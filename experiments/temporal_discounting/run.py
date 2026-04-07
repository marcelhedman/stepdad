"""
Hyperbolic temporal discounting experiment.

Example commands
-----------------------------------------------------
    # Train DAD (100K steps, T=20)
    python experiments/temporal_discounting/run.py --mode dad --n_steps 100000 --T 20

    # Step-DAD with τ=10 and 1K fine-tuning steps
    python experiments/temporal_discounting/run.py --mode stepdad --tau 10 --n_finetune_steps 1000 --T 20

    # Static baseline
    python experiments/temporal_discounting/run.py --mode static --n_steps 100000 --T 20

    # Random baseline
    python experiments/temporal_discounting/run.py --mode random --T 20
"""

import argparse
import numpy as np
import torch

from stepdad.models.temporal_discounting import TemporalDiscountingPrior, TemporalDiscountingModel
from stepdad.policy.dad import DADPolicy
from stepdad.policy.baselines import StaticDesignNetwork, RandomDesignNetwork
from stepdad.training.train import train_dad, run_stepdad
from stepdad.objectives.spce import estimate_eig
from stepdad.logging.logger import make_logger


def parse_args():
    p = argparse.ArgumentParser(description="Temporal discounting experiment")
    p.add_argument("--mode", choices=["dad", "stepdad", "static", "random"], default="dad")
    p.add_argument("--T", type=int, default=20, help="Total experiment steps")
    # Training
    p.add_argument("--n_steps", type=int, default=100_000, help="DAD pre-training steps")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--L", type=int, default=1023)
    p.add_argument("--final_L", type=int, default=100000)
    # Step-DAD
    p.add_argument("--tau", type=int, nargs="+", default=[6], help="Fine-tuning step(s) τ")
    p.add_argument("--n_finetune_steps", type=int, default=1000)
    p.add_argument("--n_posterior_samples", type=int, default=20_000)
    p.add_argument("--finetune_lr", type=float, default=5e-5)
    p.add_argument("--n_eval_batches", type=int, default=10)
    # p.add_argument("--eval_L", type=int, default=1023)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--n_thetas", type=int, default=16, help="Number of true thetas for Step-DAD evaluation")
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--logger", choices=["stdout", "wandb"], default="stdout")
    p.add_argument("--wandb_project", type=str, default="stepdad")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    prior = TemporalDiscountingPrior(device=device)

    if args.mode == "static":
        design_net = StaticDesignNetwork(design_dim=2, T=args.T).to(device)
    elif args.mode == "random":
        design_net = RandomDesignNetwork(design_dim=2).to(device)
    else:
        design_net = DADPolicy(
            design_dim=2,
            obs_dim=1,
            hidden_dim=256,
            encoding_dim=16,
        ).to(device)

    model = TemporalDiscountingModel(
        prior=prior,
        design_net=design_net,
        T=args.T,
        device=device,
    )

    logger = make_logger(
        args.logger,
        project=args.wandb_project,
        name=f"temp_disc_{args.mode}_T{args.T}",
        config=vars(args),
    )

    print(f"=== Temporal discounting | mode={args.mode} | T={args.T} ===")

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
            gradient_estimator="reinforce",
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
                print(f"\n--- theta {i+1}/{args.n_thetas}: (log_k, alpha) = {theta_true.squeeze().tolist()} ---")
                _, _, metrics = run_stepdad(
                    model=model,
                    theta_true=theta_true,
                    refinement_schedule=args.tau,
                    gradient_estimator="reinforce",
                    T=args.T,
                    n_finetune_steps=args.n_finetune_steps,
                    n_posterior_samples=args.n_posterior_samples,
                    finetune_lr=args.finetune_lr,
                    finetune_L=args.L,
                    finetune_batch_size=args.batch_size,
                    n_eval_batches=args.n_eval_batches,
                    eval_L=args.final_L,
                    eval_batch_size=args.eval_batch_size,
                    logger=logger,
                )
                all_metrics.append(metrics)

            keys = ["total_eig_stepdad_lb", "total_eig_stepdad_ub", "total_eig_no_finetune_lb", "total_eig_no_finetune_ub", "matched_dad_lb", "matched_dad_ub"]
            means = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
            stes  = {k: float(np.std([m[k] for m in all_metrics]) / np.sqrt(args.n_thetas)) for k in keys}

            print(f"\n=== Step-DAD results (mean ± SE over {args.n_thetas} thetas) ===")
            print(f"Total EIG StepDAD     lb={means['total_eig_stepdad_lb']:.4f} ± {stes['total_eig_stepdad_lb']:.4f}"
                  f"  ub={means['total_eig_stepdad_ub']:.4f} ± {stes['total_eig_stepdad_ub']:.4f}")
            print(f"Total EIG no-finetune lb={means['total_eig_no_finetune_lb']:.4f} ± {stes['total_eig_no_finetune_lb']:.4f}"
                  f"  ub={means['total_eig_no_finetune_ub']:.4f} ± {stes['total_eig_no_finetune_ub']:.4f}")
            print(f"Total EIG matched-DAD lb={means['matched_dad_lb']:.4f} ± {stes['matched_dad_lb']:.4f}"
                  f"  ub={means['matched_dad_ub']:.4f} ± {stes['matched_dad_ub']:.4f}")
            if logger:
                logger.log({**{f"mean_{k}": v for k, v in means.items()},
                            **{f"ste_{k}": v for k, v in stes.items()}}, step=args.T)

    logger.finish()


if __name__ == "__main__":
    main()
