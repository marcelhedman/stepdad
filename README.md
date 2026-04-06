# Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design

[![arXiv](https://img.shields.io/badge/arXiv-2507.14057-b31b1b.svg)](https://arxiv.org/abs/2507.14057)
[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue.svg)](https://proceedings.mlr.press/v267)

**Marcel Hedman\*, Desi R. Ivanova\*, Cong Guan, Tom Rainforth**  
*Department of Statistics, University of Oxford*  
(\* Equal contribution)

---

## Overview

Step-DAD is a **semi-amortized, policy-based** approach to Bayesian Experimental Design (BED).
Like fully amortized methods (e.g. [DAD](https://arxiv.org/abs/2103.08564)), Step-DAD trains a design
policy offline before the experiment.  Unlike them, Step-DAD **periodically refines the policy
at test time** using data gathered so far: an *infer-refine* procedure that improves both
performance and robustness.

<p align="center">
  <img src="https://arxiv.org/html/2507.14057v1/extracted/6646186/figures/overview.png" width="700"/>
</p>

Empirically, Step-DAD consistently outperforms state-of-the-art BED methods while using
substantially less computation than traditional adaptive BED.

---

## Installation

```bash
git clone https://github.com/marcelhedman/stepdad
cd stepdad
```

Create and activate a new environment with [mamba](https://mamba.readthedocs.io) (or conda):

```bash
mamba create -n stepdad python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba activate stepdad
```

For CPU-only:

```bash
mamba create -n stepdad python=3.11 pytorch cpuonly -c pytorch
mamba activate stepdad
```

Then install the package:

```bash
pip install -e .
```

---

## Quick Start

Run experiments with `--mode dad` to train the fully amortized baseline, or `--mode stepdad` to run the full Step-DAD algorithm.

### Source Location Finding

```bash
# Train DAD policy (50K steps)
python experiments/location_finding/run.py --mode dad --n_steps 50000

# Run Step-DAD with τ=6
python experiments/location_finding/run.py --mode stepdad --tau 6 --n_finetune_steps 2500

# Multiple sources
python experiments/location_finding/run.py --mode stepdad --K 2 --tau 7
```

### Hyperbolic Temporal Discounting

```bash
# Train DAD policy (100K steps)
python experiments/temporal_discounting/run.py --mode dad --n_steps 100000 --T 20

# Step-DAD with τ=10
python experiments/temporal_discounting/run.py --mode stepdad --tau 10 --n_finetune_steps 1000
```

### Constant Elasticity of Substitution

```bash
# Train DAD policy (50K steps)
python experiments/ces/run.py --mode dad --n_steps 50000

# Step-DAD with τ=5
python experiments/ces/run.py --mode stepdad --tau 5 --n_finetune_steps 10000
```

### Logging

All scripts support `--logger stdout` (default) or `--logger wandb`:

```bash
python experiments/location_finding/run.py --mode stepdad --logger wandb --wandb_project my_project
```

---

## Repository Structure

```
stepdad/
├── experiments/
│   ├── location_finding/run.py
│   ├── temporal_discounting/run.py
│   └── ces/run.py
└── src/stepdad/
    ├── models/          # Generative models (prior + likelihood)
    │   ├── location_finding.py
    │   ├── temporal_discounting.py
    │   └── ces.py
    ├── policy/
    │   └── dad.py       # DADPolicy, CESDADPolicy
    ├── objectives/
    │   └── spce.py      # sPCE lower bound + sNMC upper bound
    ├── inference/
    │   └── importance_sampling.py
    ├── training/
    │   └── train.py     # train_dad(), run_stepdad()
    └── logging/
        └── logger.py    # StdoutLogger, WandbLogger
```

---

## Algorithm

Step-DAD implements the following infer-refine procedure at each scheduled step τ:

1. **Infer** — fit the posterior `p(θ | h_τ)` via importance sampling given the history `h_τ`.
2. **Refine** — fine-tune the policy to maximise the remaining EIG `I^{h_τ}_{τ+1→T}(π)`.
3. **Deploy** — continue the experiment with the refined policy.

The remaining EIG is optimised using the sPCE lower bound (Eq. 6 of the paper), sampling θ
from the posterior approximation rather than the prior.

---

## Citation

```bibtex
@inproceedings{hedman2025stepdad,
  title     = {{Step-DAD}: Semi-Amortized Policy-Based {B}ayesian Experimental Design},
  author    = {Hedman, Marcel and Ivanova, Desi R. and Guan, Cong and Rainforth, Tom},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  year      = {2025},
  publisher = {PMLR}
}
```

---

## Acknowledgements

MH is supported by Novo Nordisk and the EPSRC Centre for Doctoral Training in Modern Statistics
and Statistical Machine Learning.  TR is supported by the UK EPSRC grant EP/Y037200/1.
