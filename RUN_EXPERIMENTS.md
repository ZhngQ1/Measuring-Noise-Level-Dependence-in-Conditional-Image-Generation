# How to Run the Experiments

## Reproducibility (seed)

All experiments use a fixed random seed for reproducibility. Default seed is **42**. Override with:

```bash
python experiment.py --seed 42
```

## Model selection (SD 1.5 vs SDXL)

The script supports two model families:

- **SD 1.5**: `--model sd15` (default). Outputs go to `results/` and `report/figures/`.
- **SDXL**: `--model sdxl`. Outputs go to `results/sdxl/` and `report/figures/sdxl/` to avoid overwriting SD 1.5.

You can also choose which suite to run:

- **text-only**: `--text-only`
- **controlnet-only**: `--controlnet-only`
- **both**: omit both flags

## 1. Run only ControlNet structural conditioning experiments

This runs the four ControlNet experiments (CIS_struct, TS_struct, SAS_struct, selective structural) **without** running the text-conditioning experiments. Use this for a quicker run when you only need structural results.

**Requirements:** `diffusers` (with ControlNet), `opencv-python`, and the same deps as below.

```bash
python experiment.py --model sd15 --controlnet-only --seed 42
```

**Outputs:**
- `results/cis_struct_data.json`
- `results/ts_struct_data.json`
- `results/sas_struct_data.json`
- `results/selective_structural_scores.json`
- `report/figures/cis_struct_curve.pdf`
- `report/figures/ts_struct.pdf`
- `report/figures/sas_struct_curves.pdf`
- `report/figures/selective_structural_grid.pdf`

## 2. Run all experiments (text + ControlNet)

Runs text-conditioning experiments (CIS, SAS, selective, TS, multi-guidance) and then ControlNet structural experiments.

```bash
python experiment.py --model sd15 --seed 42
```

## 3. Run only text-conditioning experiments (skip ControlNet)

```bash
python experiment.py --model sd15 --text-only --seed 42
```

Results go to `results/` and figures to `report/figures/`. Paths are under the project root unless you set:

- `ECE285_OUTPUT_DIR` – directory for JSON results
- `ECE285_FIGURE_DIR` – directory for PDF figures

## 4. SDXL license & download (step-by-step)

SDXL requires accepting the model license on Hugging Face before downloading.

1) Create / log into a Hugging Face account.

2) Visit the SDXL base model page and click **"Agree and access repository"**:
- `stabilityai/stable-diffusion-xl-base-1.0`

3) Create an access token (read permission) in your Hugging Face settings.

4) Log in from your terminal (PowerShell):

```bash
pip install -U huggingface_hub
huggingface-cli login
```

Paste your token when prompted. Alternatively, set it for the current session:

```powershell
$env:HUGGINGFACE_HUB_TOKEN="hf_...your_token..."
```

5) Run SDXL experiments:

- **SDXL text-only**

```bash
python experiment.py --model sdxl --text-only --seed 42
```

- **SDXL controlnet-only**

```bash
python experiment.py --model sdxl --controlnet-only --seed 42
```

- **SDXL both**

```bash
python experiment.py --model sdxl --seed 42
```

### SDXL speed knobs (recommended for 11GB GPUs like 1080 Ti)

SDXL is significantly heavier than SD 1.5. The script exposes two SDXL-only knobs:

- `--sdxl-steps`: reduce denoising steps (default: 30)
- `--sdxl-eval-every`: evaluate/decode for SAS every N steps (default: 10)

Examples:

```bash
# Faster SDXL text-only run
python experiment.py --model sdxl --text-only --seed 42 --sdxl-steps 25 --sdxl-eval-every 10 --sdxl-vae-fp32

# Even faster (coarser SAS curve)
python experiment.py --model sdxl --text-only --seed 42 --sdxl-steps 20 --sdxl-eval-every 20 --sdxl-vae-fp32
```

Notes:
- SDXL downloads are large and require substantial GPU VRAM; reduce `IMG_HEIGHT/IMG_WIDTH` if needed.
- SDXL ControlNet defaults to `diffusers/controlnet-canny-sdxl-1.0` (override via `SDXL_CONTROLNET_MODEL_ID`).

## 5. Environment

- **GPU:** Recommended (CUDA). CPU will be slow.
- **Models:** Script loads from **Hugging Face cache** by default. If you already downloaded SD 1.5 (e.g. under `C:\Users\<你>\.cache\huggingface\hub`), it will use the cache and not re-download.
  - SD 1.5: `runwayml/stable-diffusion-v1-5` (cached as `models--runwayml--stable-diffusion-v1-5`)
  - ControlNet Canny: `lllyasviel/control_v11p_sd15_canny` (cached similarly)
- **Custom SD 1.5 path:** If you use a local folder instead of cache, set env and run:
  ```bash
  set SD15_MODEL_ID=你的本地路径\stable-diffusion-v1-5
  python experiment.py --controlnet-only --seed 42
  ```

Install deps:

```bash
pip install -r requirements.txt
```

## 6. After running

Share the contents of `results/` (all JSON files) and `report/figures/` (all PDFs) so the report can be updated with the new results.
