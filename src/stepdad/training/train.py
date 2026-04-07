"""
DAD training and Step-DAD online infer-refine loop.

Two public functions:

1. train_dad(model, n_steps, ...)
   Standard offline DAD training: simulate trajectories, compute sPCE loss,
   update ``model.design_net`` via Adam.  Accepts an optional ``final_L`` for
   a high-quality EIG estimate on a fresh batch at the last step.

2. run_stepdad(model, theta_true, refinement_schedule, ...)
   Online Step-DAD deployment:
   - Uses ``run_model.design_net`` (the working copy's policy) to collect
     designs until τ.
   - Fits a posterior via IS given the observed history h_τ.
   - Fine-tunes ``run_model.design_net`` on the remaining EIG objective by
     wrapping ``run_model`` in a _PosteriorModel and calling train_dad.
   - Evaluates EIG(τ→T) for both the fine-tuned and original (matched-DAD)
     policies, averaged over ``n_eval_batches`` rollouts.
   - Evaluates EIG(0→τ) using a fresh copy of the original model.
   - Returns (designs, outcomes, eval_metrics).

Both functions accept an optional Logger for metrics.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange

from stepdad.objectives.spce import SPCELoss, estimate_eig
from stepdad.inference.importance_sampling import importance_sample, resample
from stepdad.logging.logger import Logger


# ---------------------------------------------------------------------------
# Offline DAD training
# ---------------------------------------------------------------------------

def train_dad(
    model: nn.Module,
    n_steps: int,
    gradient_estimator: str,
    batch_size: int = 1024,
    L: int = 1023,
    lr: float = 1e-4,
    log_every: int = 500,
    final_L: int | None = None,
    logger: Logger | None = None,
    past_designs: Tensor | None = None,
    past_outcomes: Tensor | None = None,
) -> None:
    """Train the policy offline via the sPCE lower bound.

    The policy is ``model.design_net``; gradients flow through it during
    training and the caller accesses the trained policy as ``model.design_net``
    after this call returns.

    Args:
        model:          generative model with a ``design_net`` attribute.
        n_steps:        number of gradient steps.
        batch_size:     trajectories per gradient step (B).
        L:              contrastive samples for sPCE loss (default 1023).
        lr:             Adam learning rate (default 1e-4).
        log_every:      evaluation / logging interval (steps).
        final_L:        if provided, run a separate fresh-batch EIG estimate at
                        the last step with this many contrastive samples for a
                        tighter final evaluation.
        logger:         optional Logger for metrics.
        past_designs:   [B, tau, p]  — fixed observed history to condition on
                        (used for fine-tuning at step τ in Step-DAD).
        past_outcomes:  [B, tau, obs_dim]
    """
    optimiser = torch.optim.Adam(model.design_net.parameters(), lr=lr)
    spce_loss = SPCELoss(model, gradient_estimator=gradient_estimator, L=L, lower_bound=True)

    for step in trange(1, n_steps + 1, desc="DAD training"):
        optimiser.zero_grad()

        theta, designs, outcomes = model(
            batch_size,
            past_designs=past_designs,
            past_outcomes=past_outcomes,
        )

        loss = spce_loss(theta, designs, outcomes)
        loss.backward()
        optimiser.step()

        # Log loss at every step
        if logger:
            logger.log({"loss": loss.item()}, step=step)

        if step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                eig_lb = estimate_eig(model, theta, designs, outcomes, L=1023, lower_bound=True)
                eig_ub = estimate_eig(model, theta, designs, outcomes, L=1023, lower_bound=False)
            eig_metrics = {"eig_lower": eig_lb, "eig_upper": eig_ub}
            if logger:
                logger.log(eig_metrics, step=step)
            parts = [f"step={step}", f"loss={loss.item():.4f}"] + [f"{k}={v:.4f}" for k, v in eig_metrics.items()]
            print("  ".join(parts))

        # Separate high-quality final evaluation with larger L on a fresh batch
        if step == n_steps and final_L is not None and final_L != L:
            with torch.no_grad():
                theta_eval, designs_eval, outcomes_eval = model(
                    batch_size,
                    past_designs=past_designs,
                    past_outcomes=past_outcomes,
                )
                final_lb = estimate_eig(
                    model, theta_eval, designs_eval, outcomes_eval,
                    L=final_L, lower_bound=True,
                )
                final_ub = estimate_eig(
                    model, theta_eval, designs_eval, outcomes_eval,
                    L=final_L, lower_bound=False,
                )
            final_metrics = {"final_eig_lower": final_lb, "final_eig_upper": final_ub}
            if logger:
                logger.log(final_metrics, step=step)
            parts = [f"step={step} [final L={final_L}]"] + [f"{k}={v:.4f}" for k, v in final_metrics.items()]
            print("  ".join(parts))


# ---------------------------------------------------------------------------
# Online Step-DAD infer-refine loop
# ---------------------------------------------------------------------------

def run_stepdad(
    model: nn.Module,
    theta_true: Tensor,
    refinement_schedule: list[int],
    gradient_estimator: str,
    T: int | None = None,
    n_finetune_steps: int = 2500,
    n_posterior_samples: int = 20_000,
    finetune_lr: float = 1e-4,
    finetune_L: int = 1023,
    finetune_batch_size: int = 16,
    n_eval_batches: int = 10,
    eval_L: int = 1023,
    eval_batch_size: int = 512,
    logger: Logger | None = None,
) -> tuple[Tensor, Tensor, dict]:
    """Run the Step-DAD online infer-refine loop for a single experimental instance.

    At each τ in the refinement schedule, we:
      1. Fit the posterior p(θ | h_τ) via importance sampling.
      2. Fine-tune ``run_model.design_net`` on the remaining EIG objective by
         wrapping ``run_model`` in a _PosteriorModel and calling train_dad.
         Because _PosteriorModel.design_net mirrors run_model.design_net, the
         fine-tuned weights land directly on run_model.design_net — no copy-back.
      3. Evaluate EIG(τ→T) for:
           - fine-tuned policy (run_model copy, updated design_net)
           - original DAD policy  (fresh copy of the caller's model)
         averaged over ``n_eval_batches`` rollouts.

    After all refinements, EIG(0→τ) is estimated using a fresh copy of the
    caller's model (original policy) with T=τ.

    Total EIG = EIG(0→τ) + EIG(τ→T) via sequential MI additivity.

    Args:
        model:                 generative model; model.design_net is the pre-trained policy.
        theta_true:            [1, theta_dim] — ground-truth θ for this instance.
        refinement_schedule:   sorted list of τ values at which to refine, e.g. [6].
        gradient_estimator:    "rparam" or "reinforce"
        T:                     total experiment steps; defaults to model.T.
        n_finetune_steps:      gradient steps for each fine-tuning (default 2500).
        n_posterior_samples:   IS samples for posterior (default 20 000).
        finetune_lr:           fine-tuning learning rate (default 1e-4).
        finetune_L:            contrastive samples during fine-tuning (default 1023).
        finetune_batch_size:   trajectories per fine-tune gradient step (default 16).
        n_eval_batches:        outer averaging iterations for EIG estimation at τ
                               (default 10).
        eval_L:                contrastive samples per EIG evaluation batch (default 1023).
        eval_batch_size:       trajectories per evaluation batch (default 512).
        logger:                optional Logger.

    Returns:
        designs:      [1, T, design_dim]       — full history of designs.
        outcomes:     [1, T, obs_dim] — full history of outcomes.
        eval_metrics: dict containing EIG estimates:
            "eig_from_tau":    {"finetuned_lb/ub", "notuning_lb/ub"}
            "eig_upto_tau_lb/ub"
            "total_eig_stepdad_lb/ub"
            "total_eig_no_finetune_lb/ub"
    """
    if T is None:
        T = model.T

    # All work is done on a copy so the caller's model is never mutated.
    # model (caller's copy) always holds the pre-trained design_net throughout.
    run_model = copy.deepcopy(model)

    device = theta_true.device
    designs = torch.empty(1, 0, _design_dim(run_model), device=device)
    outcomes = torch.empty(1, 0, 1, device=device)

    tau_sorted = sorted(refinement_schedule)
    tau_set = set(tau_sorted)
    t = 0

    # Per-segment EIG estimates — summed to give total EIG from tau onwards.
    # Segment at τ covers [τ, next_τ) so they tile without overlap.
    _ft_lb, _ft_ub, _nt_lb, _nt_ub = [], [], [], []

    def _segment_end(current_t: int) -> int:
        """Return the end of the eval segment starting at current_t."""
        for tau in tau_sorted:
            if tau > current_t:
                return tau
        return T

    while t < T:
        # --- collect one step with current policy ---
        with torch.no_grad():
            xi = run_model.design_net(designs, outcomes)   # [1, d] raw
            xi_stored = run_model.transform_design(xi)     # constrained for CES, identity otherwise
            yi = model.outcome_likelihood(
                theta_true, xi
            ).sample()                                     # [1, obs_dim]
            if yi.dim() == 1:
                yi = yi.unsqueeze(-1)

        designs = torch.cat([designs, xi_stored.unsqueeze(1)], dim=1)
        outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1)
        t += 1

        if logger:
            logger.log({"step": t}, step=t)

        # --- infer-refine at scheduled τ values ---
        if t in tau_set and t < T:
            _log(logger, t, f"Step-DAD: posterior inference at τ={t}")

            # Infer: IS posterior given h_τ
            theta_samples, log_weights, ESS = importance_sample(
                run_model, designs, outcomes,
                n_samples=n_posterior_samples,
                device=device,
            )
            _log(logger, t, f"  ESS = {ESS:.0f} / {n_posterior_samples}")
            if logger:
                logger.log({"ESS": ESS}, step=t)

            # Refine: fine-tune run_model.design_net on remaining EIG objective.
            # _PosteriorModel stores run_model directly; train_dad updates
            # run_model.design_net in-place through posterior_model.design_net.
            _log(logger, t, f"  Fine-tuning for {n_finetune_steps} steps ...")

            post_samples = resample(theta_samples, log_weights, n_posterior_samples)

            posterior_model = _PosteriorModel(
                model=run_model,
                posterior_samples=post_samples,
                remaining_T=T - t,
                past_designs=designs.expand(finetune_batch_size, -1, -1),
                past_outcomes=outcomes.expand(finetune_batch_size, -1, -1),
            )
            train_dad(
                model=posterior_model,
                n_steps=n_finetune_steps,
                gradient_estimator=gradient_estimator,
                batch_size=finetune_batch_size,
                L=finetune_L,
                lr=finetune_lr,
                log_every=max(1, n_finetune_steps // 10),
                logger=logger,
            )
            # run_model.design_net is now fine-tuned (updated in-place via posterior_model)

            # ------------------------------------------------------------------
            # Evaluation: EIG(τ → next_τ) for fine-tuned vs original (matched-DAD).
            # Each segment covers only until the next refinement point (or T),
            # so segments tile without overlap and can be summed for total EIG.
            # Fine-tuning above uses T-t (full horizon); eval uses the shorter segment.
            # ------------------------------------------------------------------
            seg_end = _segment_end(t)
            seg_len = seg_end - t
            _log(logger, t, f"  Evaluating EIG(τ={t}→{seg_end}) over {n_eval_batches} batches ...")


            eval_post = resample(theta_samples, log_weights, n_posterior_samples)

            # Fine-tuned eval: isolated copy of run_model (has fine-tuned design_net)
            eval_ft_model = _PosteriorModel(
                model=copy.deepcopy(run_model),
                posterior_samples=eval_post,
                remaining_T=seg_len,
                past_designs=designs.expand(eval_batch_size, -1, -1),
                past_outcomes=outcomes.expand(eval_batch_size, -1, -1),
            )
            # No-tuning eval: fresh copy of original model (pre-trained design_net, never mutated)
            eval_nt_model = _PosteriorModel(
                model=copy.deepcopy(model),
                posterior_samples=eval_post,
                remaining_T=seg_len,
                past_designs=designs.expand(eval_batch_size, -1, -1),
                past_outcomes=outcomes.expand(eval_batch_size, -1, -1),
            )

            ft_lb_vals, ft_ub_vals, nt_lb_vals, nt_ub_vals = [], [], [], []
            with torch.no_grad():
                for _ in range(n_eval_batches):
                    theta_ft, d_ft, o_ft = eval_ft_model(eval_batch_size)
                    theta_nt, d_nt, o_nt = eval_nt_model(eval_batch_size)
                    ft_lb_vals.append(estimate_eig(eval_ft_model, theta_ft, d_ft, o_ft, L=eval_L, lower_bound=True))
                    ft_ub_vals.append(estimate_eig(eval_ft_model, theta_ft, d_ft, o_ft, L=eval_L, lower_bound=False))
                    nt_lb_vals.append(estimate_eig(eval_nt_model, theta_nt, d_nt, o_nt, L=eval_L, lower_bound=True))
                    nt_ub_vals.append(estimate_eig(eval_nt_model, theta_nt, d_nt, o_nt, L=eval_L, lower_bound=False))

            ft_lb = float(torch.tensor(ft_lb_vals).mean())
            ft_ub = float(torch.tensor(ft_ub_vals).mean())
            nt_lb = float(torch.tensor(nt_lb_vals).mean())
            nt_ub = float(torch.tensor(nt_ub_vals).mean())

            _ft_lb.append(ft_lb); _ft_ub.append(ft_ub)
            _nt_lb.append(nt_lb); _nt_ub.append(nt_ub)

            _log(logger, t, f"  EIG({t}→{seg_end}) finetuned  lb={ft_lb:.4f}  ub={ft_ub:.4f}")
            _log(logger, t, f"  EIG({t}→{seg_end}) no-tuning  lb={nt_lb:.4f}  ub={nt_ub:.4f}")
            if logger:
                logger.log({
                    "eig_from_tau_ft_lb": ft_lb,
                    "eig_from_tau_ft_ub": ft_ub,
                    "eig_from_tau_nt_lb": nt_lb,
                    "eig_from_tau_nt_ub": nt_ub,
                }, step=t)

    # ------------------------------------------------------------------
    # EIG(0→τ₁): fresh copy of original model (pre-trained policy, T=first τ)
    # model is untouched so its design_net is always the pre-trained policy.
    # ------------------------------------------------------------------
    tau = tau_sorted[0]
    upto_model = copy.deepcopy(model)
    upto_model.T = tau

    # Matched-DAD: pre-trained policy on theta_true for all T steps.
    # Each rollout is independent (stochastic outcomes) but shares theta_true,
    # giving a direct like-for-like comparison with StepDAD on this instance.
    matched_dad_model = copy.deepcopy(model)
    theta_matched = theta_true.expand(eval_batch_size, -1)

    upto_lb_vals, upto_ub_vals = [], []
    matched_dad_lb_vals, matched_dad_ub_vals = [], []
    with torch.no_grad():
        for _ in range(n_eval_batches):
            theta_upto, designs_upto, outcomes_upto = upto_model(eval_batch_size)
            upto_lb_vals.append(estimate_eig(upto_model, theta_upto, designs_upto, outcomes_upto, L=eval_L, lower_bound=True))
            upto_ub_vals.append(estimate_eig(upto_model, theta_upto, designs_upto, outcomes_upto, L=eval_L, lower_bound=False))

            designs_matched, outcomes_matched = matched_dad_model.run_policy(theta_matched)
            matched_dad_lb_vals.append(estimate_eig(matched_dad_model, theta_matched, designs_matched, outcomes_matched, L=eval_L, lower_bound=True))
            matched_dad_ub_vals.append(estimate_eig(matched_dad_model, theta_matched, designs_matched, outcomes_matched, L=eval_L, lower_bound=False))

    upto_lb = float(torch.tensor(upto_lb_vals).mean())
    upto_ub = float(torch.tensor(upto_ub_vals).mean())
    matched_dad_lb = float(torch.tensor(matched_dad_lb_vals).mean())
    matched_dad_ub = float(torch.tensor(matched_dad_ub_vals).mean())

    # Sum per-segment EIG estimates — each segment tiles [τ_i, τ_{i+1}) without overlap
    eig_ft_lb = float(sum(_ft_lb))
    eig_ft_ub = float(sum(_ft_ub))
    eig_nt_lb = float(sum(_nt_lb))
    eig_nt_ub = float(sum(_nt_ub))

    total_stepdad_lb = upto_lb + eig_ft_lb
    total_stepdad_ub = upto_ub + eig_ft_ub
    total_no_finetune_lb = upto_lb + eig_nt_lb
    total_no_finetune_ub = upto_ub + eig_nt_ub

    _log(logger, T, f"EIG(0→τ)  lb={upto_lb:.4f}  ub={upto_ub:.4f}")
    _log(logger, T, f"Total EIG  StepDAD     lb={total_stepdad_lb:.4f}  ub={total_stepdad_ub:.4f}")
    _log(logger, T, f"Total EIG  no-finetune lb={total_no_finetune_lb:.4f}  ub={total_no_finetune_ub:.4f}")
    _log(logger, T, f"Total EIG  matched-DAD lb={matched_dad_lb:.4f}  ub={matched_dad_ub:.4f}")
    if logger:
        logger.log({
            "eig_upto_tau_lb": upto_lb,
            "eig_upto_tau_ub": upto_ub,
            "total_eig_stepdad_lb": total_stepdad_lb,
            "total_eig_stepdad_ub": total_stepdad_ub,
            "total_eig_no_finetune_lb": total_no_finetune_lb,
            "total_eig_no_finetune_ub": total_no_finetune_ub,
            "matched_dad_lb": matched_dad_lb,
            "matched_dad_ub": matched_dad_ub,
        }, step=T)

    eval_metrics = {
        "eig_from_tau": {
            "finetuned_lb": eig_ft_lb,
            "finetuned_ub": eig_ft_ub,
            "notuning_lb": eig_nt_lb,
            "notuning_ub": eig_nt_ub,
        },
        "eig_upto_tau_lb": upto_lb,
        "eig_upto_tau_ub": upto_ub,
        "total_eig_stepdad_lb": total_stepdad_lb,
        "total_eig_stepdad_ub": total_stepdad_ub,
        "total_eig_no_finetune_lb": total_no_finetune_lb,
        "total_eig_no_finetune_ub": total_no_finetune_ub,
        "matched_dad_lb": matched_dad_lb,
        "matched_dad_ub": matched_dad_ub,
    }

    return designs, outcomes, eval_metrics


# ---------------------------------------------------------------------------
# Helper: thin posterior model wrapper for fine-tuning and evaluation
# ---------------------------------------------------------------------------

class _PosteriorModel(nn.Module):
    """A model wrapper that samples theta from a fixed posterior approximation.

    Used during Step-DAD fine-tuning and evaluation to condition on the
    observed history h_τ and optimise/evaluate the remaining EIG
    I^{h_τ}_{τ+1→T}.

    The policy is ``model.design_net``.
    The caller is responsible for passing an appropriately isolated model copy
    when mutation isolation is required (see run_stepdad for usage patterns).
    """

    def __init__(
        self,
        model: nn.Module,
        posterior_samples: Tensor,
        remaining_T: int,
        past_designs: Tensor,
        past_outcomes: Tensor,
    ) -> None:
        super().__init__()
        self._model = model
        self.register_buffer("_posterior_samples", posterior_samples)
        self._remaining_T = remaining_T
        self.register_buffer("_past_designs", past_designs)
        self.register_buffer("_past_outcomes", past_outcomes)
        # Expose prior and log_likelihood for SPCELoss compatibility
        self.prior = _PosteriorPrior(posterior_samples)
        self.design_dim = _design_dim(model)

    @property
    def design_net(self) -> nn.Module:
        return self._model.design_net

    @design_net.setter
    def design_net(self, net: nn.Module) -> None:
        self._model.design_net = net

    @property
    def T(self) -> int:
        return self._remaining_T

    def log_likelihood(self, theta: Tensor, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Delegate to underlying model."""
        return self._model.log_likelihood(theta, designs, outcomes)

    def forward(self, batch_size: int, past_designs=None, past_outcomes=None):
        """Sample theta from the posterior and simulate remaining steps."""
        # Resample from posterior for this batch
        idx = torch.randint(0, self._posterior_samples.shape[0], (batch_size,))
        theta = self._posterior_samples[idx]

        # Use the already-observed history as the starting point
        pd = self._past_designs[:batch_size]
        po = self._past_outcomes[:batch_size]

        designs_full = pd.clone()
        outcomes_full = po.clone()

        for _ in range(self._remaining_T):
            xi = self._model.design_net(designs_full, outcomes_full)          # raw
            xi_stored = self._model.transform_design(xi)                       # constrained for CES
            lk = self._model.outcome_likelihood(theta, xi)
            yi = lk.rsample() if lk.has_rsample else lk.sample()              # rsample for rparam, sample for reinforce
            if yi.dim() == 1:
                yi = yi.unsqueeze(-1)
            designs_full = torch.cat([designs_full, xi_stored.unsqueeze(1)], dim=1)
            outcomes_full = torch.cat([outcomes_full, yi.unsqueeze(1)], dim=1)

        # Return only the newly simulated steps
        new_d = designs_full[:, pd.shape[1]:]
        new_o = outcomes_full[:, po.shape[1]:]
        return theta, new_d, new_o

    def outcome_likelihood(self, theta, xi):
        return self._model.outcome_likelihood(theta, xi)


class _PosteriorPrior:
    """A 'prior' that samples from a fixed set of posterior samples."""

    def __init__(self, samples: Tensor) -> None:
        self._samples = samples

    def sample(self, n: int) -> Tensor:
        idx = torch.randint(0, self._samples.shape[0], (n,))
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log(logger: Logger | None, step: int, msg: str) -> None:  # noqa: ARG001
    if logger is None:
        print(msg)


def _design_dim(model: nn.Module) -> int:
    """Infer design dimension from the model."""
    for attr in ("p", "design_dim"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError("Cannot infer design_dim from model.")