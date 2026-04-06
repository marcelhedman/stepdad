"""
Location finding experiment.

Modes
-----
    dad        Train a DAD policy offline for T=10 steps.
    stepdad    Pre-train DAD then run the Step-DAD infer-refine loop.
    static     Learn a fixed set of T designs (non-adaptive baseline).
    random     Evaluate a random design baseline.

Example commands
-----------------------------------------------------
    # Train DAD (50K steps, single source)
    python experiments/location_finding/run.py --mode dad --n_steps 50000

    # Step-DAD with τ=7 and 2.5K fine-tuning steps
    python experiments/location_finding/run.py --mode stepdad --tau 7 --n_finetune_steps 2500

    # Multiple sources (4-dim theta)
    python experiments/location_finding/run.py --mode stepdad --K 2 --tau 7

    # Static baseline
    python experiments/location_finding/run.py --mode static --n_steps 50000

    # Random baseline
    python experiments/location_finding/run.py --mode random

    # Log to W&B
    python experiments/location_finding/run.py --mode stepdad --logger wandb --wandb_project stepdad
"""

import argparse
import numpy as np
import torch

from stepdad.models.location_finding import LocationFindingPrior, LocationFindingModel
from stepdad.policy.dad import DADPolicy
from stepdad.policy.baselines import StaticDesignNetwork, RandomDesignNetwork
from stepdad.training.train import train_dad, run_stepdad
from stepdad.objectives.spce import estimate_eig
from stepdad.logging.logger import make_logger


def parse_args():
    p = argparse.ArgumentParser(description="Location finding experiment")
    p.add_argument("--mode", choices=["dad", "stepdad", "static", "random"], default="dad")
    p.add_argument("--T", type=int, default=10, help="Total experiment steps")
    p.add_argument("--K", type=int, default=1, help="Number of sources")
    p.add_argument("--p", type=int, default=2, help="Spatial dimension")
    # Training
    p.add_argument("--n_steps", type=int, default=50_000, help="DAD pre-training steps")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--L", type=int, default=1023, help="Contrastive samples during training")
    p.add_argument("--final_L", type=int, default=100000, help="Contrastive samples for final eval")
    # Step-DAD
    p.add_argument("--tau", type=int, nargs="+", default=[6], help="Fine-tuning step(s) τ")
    p.add_argument("--n_finetune_steps", type=int, default=2500)
    p.add_argument("--n_posterior_samples", type=int, default=20_000)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--n_eval_batches", type=int, default=10)
    # p.add_argument("--eval_L", type=int, default=1023)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--n_thetas", type=int, default=16, help="Number of true thetas for Step-DAD evaluation")
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logger", choices=["stdout", "wandb"], default="stdout")
    p.add_argument("--wandb_project", type=str, default="stepdad")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior = LocationFindingPrior(K=args.K, p=args.p, device=device)

    if args.mode == "static":
        design_net = StaticDesignNetwork(design_dim=args.p, T=args.T).to(device)
    elif args.mode == "random":
        design_net = RandomDesignNetwork(design_dim=args.p).to(device)
    else:
        design_net = DADPolicy(
            design_dim=args.p,
            obs_dim=1,
            hidden_dim=64,
            encoding_dim=16,
        ).to(device)

    model = LocationFindingModel(
        prior=prior,
        design_net=design_net,
        T=args.T,
        device=device,
    )

    logger = make_logger(
        args.logger,
        project=args.wandb_project,
        name=f"loc_finding_{args.mode}_K{args.K}_T{args.T}",
        config=vars(args),
    )

    print(f"=== Location finding | mode={args.mode} | K={args.K} | T={args.T} ===")

    if args.mode == "random":
        print("Random baseline: evaluating EIG ...")
        with torch.no_grad():
            theta_eval, d_eval, o_eval = model(args.batch_size)
            eig_lb = estimate_eig(model, theta_eval, d_eval, o_eval, L=args.final_L, lower_bound=True)
            eig_ub = estimate_eig(model, theta_eval, d_eval, o_eval, L=args.final_L, lower_bound=False)
        print(f"Random EIG  lb={eig_lb:.4f}  ub={eig_ub:.4f}")
        if logger:
            logger.log({"eig_lower": eig_lb, "eig_upper": eig_ub}, step=0)
    else:
        print(f"Pre-training {'static' if args.mode == 'static' else 'DAD'} for {args.n_steps} steps ...")
        train_dad(
            model=model,
            n_steps=args.n_steps,
            gradient_estimator="rparam",
            batch_size=args.batch_size,
            L=args.L,
            lr=args.lr,
            log_every=max(1, args.n_steps // 20),
            final_L=args.final_L,
            logger=logger,
        )

        if args.mode == "stepdad":
            print(f"\nRunning Step-DAD with τ={args.tau}, {args.n_finetune_steps} fine-tune steps, {args.n_thetas} true thetas ...")
            all_metrics = []
            for i in range(args.n_thetas):
                theta_true = prior.sample(1).to(device)
                print(f"\n--- theta {i+1}/{args.n_thetas}: {theta_true.squeeze().tolist()} ---")
                _, _, metrics = run_stepdad(
                    model=model,
                    theta_true=theta_true,
                    refinement_schedule=args.tau,
                    gradient_estimator="rparam",
                    T=args.T,
                    n_finetune_steps=args.n_finetune_steps,
                    n_posterior_samples=args.n_posterior_samples,
                    finetune_lr=args.finetune_lr,
                    finetune_L=args.L,
                    n_eval_batches=args.n_eval_batches,
                    eval_L=args.final_L,
                    eval_batch_size=args.eval_batch_size,
                    logger=logger,
                )
                all_metrics.append(metrics)

            keys = ["total_eig_stepdad_lb", "total_eig_stepdad_ub", "total_eig_dad_lb", "total_eig_dad_ub"]
            means = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
            stes  = {k: float(np.std([m[k] for m in all_metrics]) / np.sqrt(args.n_thetas)) for k in keys}

            print(f"\n=== Step-DAD results (mean ± SE over {args.n_thetas} thetas) ===")
            print(f"Total EIG StepDAD  lb={means['total_eig_stepdad_lb']:.4f} ± {stes['total_eig_stepdad_lb']:.4f}"
                  f"  ub={means['total_eig_stepdad_ub']:.4f} ± {stes['total_eig_stepdad_ub']:.4f}")
            print(f"Total EIG DAD      lb={means['total_eig_dad_lb']:.4f} ± {stes['total_eig_dad_lb']:.4f}"
                  f"  ub={means['total_eig_dad_ub']:.4f} ± {stes['total_eig_dad_ub']:.4f}")
            if logger:
                logger.log({**{f"mean_{k}": v for k, v in means.items()},
                            **{f"ste_{k}": v for k, v in stes.items()}}, step=args.T)

    logger.finish()


if __name__ == "__main__":
    main()
