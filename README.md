# Measuring Noise-Level Dependence in Conditional Image Generation

We study how the effectiveness of conditioning in diffusion models depends on noise level during inference. Through controlled inference-time interventions and quantitative diagnostics, we reveal that **conditioning is not equally influential at all noise levels** — early and mid-stage conditioning disproportionately shapes global structure and semantic content, while late-stage conditioning primarily refines details.

> **Authors:** Qile Zhang, Wuxi Chen — UC San Diego  
> **Course:** ECE 285, UC San Diego  
> **Status:** Extended research in progress; exploring potential arXiv submission.

## Motivation

Conditional diffusion models typically apply conditioning **uniformly** across all reverse diffusion steps. But is this necessary? The reverse process spans qualitatively different regimes — from coarse structure formation (high noise) to fine detail refinement (low noise). Understanding *when* conditioning matters most has both scientific and practical value for more efficient inference.

## Proposed Metrics

We define three complementary metrics to quantify conditioning effectiveness at each timestep:

| Metric | What it measures |
|--------|-----------------|
| **Conditioning Influence Score (CIS)** | L2 norm of the conditioning direction (conditional − unconditional noise prediction) at each step |
| **Trajectory Sensitivity (TS)** | Latent divergence between denoising trajectories of semantically related prompt pairs sharing initial noise |
| **Semantic Alignment Score (SAS)** | CLIP similarity between the predicted clean image at each step and the text prompt |

## Key Findings

- **CIS peaks in the early-to-mid denoising phase**, indicating conditioning exerts its strongest instantaneous influence at intermediate noise levels. Higher guidance scales amplify CIS uniformly but preserve this temporal profile.
- **Trajectory divergence accumulates most rapidly during steps 5–25**, confirming that conditioning-induced structural decisions are made early.
- **Semantic alignment emerges rapidly in early steps and plateaus by mid phase** — prompt-aligned content crystallizes before fine details are added.
- **Selective conditioning experiments** show that mid-phase conditioning alone (CLIP score 0.194) outperforms both early-only (0.166) and late-only (0.153), while full conditioning achieves 0.255.

## Supported Models

| Model | Flag | Notes |
|-------|------|-------|
| Stable Diffusion 1.5 | `--model sd15` (default) | Text + ControlNet conditioning |
| SDXL | `--model sdxl` | Text + ControlNet; requires HF license acceptance |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Requirements:** PyTorch with CUDA, `diffusers`, `transformers`, `opencv-python`, `clip` (ViT-B/32)

### Run Experiments

```bash
# All experiments (text + ControlNet), SD 1.5
python experiment.py --model sd15 --seed 42

# Text-conditioning only
python experiment.py --model sd15 --text-only --seed 42

# ControlNet structural conditioning only
python experiment.py --model sd15 --controlnet-only --seed 42

# SDXL (requires HF token, see RUN_EXPERIMENTS.md)
python experiment.py --model sdxl --text-only --seed 42
```

See [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) for full instructions including SDXL setup, speed tuning, and custom model paths.

## Repository Structure

```
├── experiment.py           # Main experiment script (all metrics + interventions)
├── RUN_EXPERIMENTS.md      # Detailed instructions for reproducing experiments
├── report/                 # Project report (NeurIPS-style LaTeX)
│   └── figures/            # Generated figures (CIS curves, TS plots, SAS curves, etc.)
├── results/                # JSON output files from experiments
│   └── sdxl/               # SDXL-specific results
└── README.md
```

## Example Results

### Conditioning Influence Score
CIS across 50 DDIM denoising steps, averaged over 8 prompts. The score peaks in the early-to-mid phase, indicating conditioning has the strongest effect at intermediate noise levels.

### Selective Conditioning
| Condition | Avg CLIP Score |
|-----------|---------------|
| Full conditioning | **0.255** |
| Mid-phase only | 0.194 |
| Early-phase only | 0.166 |
| Late-phase only | 0.153 |
| No conditioning | 0.094 |

## Practical Implications

These findings suggest that **selectively disabling conditioning at low-noise steps** could reduce computational cost (by skipping the unconditional forward pass in CFG) without significantly sacrificing generation quality — a lightweight inference-time optimization requiring no retraining.

## Citation

```bibtex
@misc{zhang2025noise,
  title={Measuring Noise-Level Dependence in Conditional Image Generation},
  author={Zhang, Qile and Chen, Wuxi},
  year={2025},
  note={UC San Diego, ECE 285 Course Project}
}
```

## Acknowledgments

This project was developed for ECE 285 at UC San Diego. We thank the course instructors for their guidance.
