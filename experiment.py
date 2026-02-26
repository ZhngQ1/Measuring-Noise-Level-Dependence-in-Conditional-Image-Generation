"""
Measuring Noise-Level Dependence in Conditional Image Generation
Main experiment script: custom DDIM sampling with CIS, TS, and SAS metrics.
Uses Stable Diffusion 1.5 via HuggingFace diffusers.
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
OUTPUT_DIR = "/mnt/root/charles/ECE285_project/results"
FIGURE_DIR = "/mnt/root/charles/ECE285_project/report/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMG_HEIGHT = 512
IMG_WIDTH = 512

PROMPTS = [
    "A photograph of a cat sitting on a windowsill",
    "A red sports car on a mountain road",
    "A watercolor painting of a sunset over the ocean",
    "An astronaut riding a horse on the moon",
    "A bowl of fruit on a wooden table",
    "A medieval castle in a foggy forest",
    "A portrait of an elderly man with glasses",
    "A snowy mountain landscape with pine trees",
]

PROMPT_PAIRS = [
    ("A red sports car on a mountain road",
     "A blue sports car on a mountain road"),
    ("A photograph of a cat sitting on a windowsill",
     "A photograph of a dog sitting on a windowsill"),
    ("A medieval castle in a foggy forest",
     "A medieval castle in a sunny meadow"),
    ("A snowy mountain landscape with pine trees",
     "A snowy mountain landscape with palm trees"),
]


def load_sd_pipeline():
    scheduler = DDIMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def encode_prompt(pipe, prompt, negative_prompt=""):
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(DEVICE))[0]
    uncond_inputs = pipe.tokenizer(
        negative_prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(DEVICE))[0]
    return text_embeddings, uncond_embeddings


def predict_noise(pipe, latents, t, text_embeddings, uncond_embeddings):
    latent_model_input = pipe.scheduler.scale_model_input(latents, t)
    with torch.no_grad():
        noise_pred_uncond = pipe.unet(
            latent_model_input, t, encoder_hidden_states=uncond_embeddings
        ).sample
        noise_pred_cond = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample
    return noise_pred_cond, noise_pred_uncond


def ddim_step(pipe, latents, t, noise_pred, t_prev):
    alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
    if t_prev >= 0:
        alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[t_prev]
    else:
        alpha_prod_t_prev = pipe.scheduler.final_alpha_cumprod

    pred_x0 = (latents - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
    pred_x0 = pred_x0.clamp(-1, 1)
    dir_xt = (1 - alpha_prod_t_prev).sqrt() * noise_pred
    prev_latents = alpha_prod_t_prev.sqrt() * pred_x0 + dir_xt
    return prev_latents, pred_x0


def predicted_x0_to_image(pipe, pred_x0):
    pred_x0_scaled = pred_x0 / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(pred_x0_scaled.to(DTYPE)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype(np.uint8)
    return Image.fromarray(image[0])


def compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, image, text):
    img_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    text_tokens = clip_tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        img_features = clip_model.encode_image(img_tensor)
        txt_features = clip_model.encode_text(text_tokens)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        similarity = (img_features @ txt_features.T).item()
    return similarity


def run_cis_and_sas(pipe, clip_model, clip_preprocess, clip_tokenizer, prompts, seed=SEED):
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    all_cis = []
    all_sas = []

    for pi, prompt in enumerate(prompts):
        print(f"  [CIS+SAS] Prompt {pi+1}/{len(prompts)}: {prompt[:50]}...")
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        text_emb, uncond_emb = encode_prompt(pipe, prompt)
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        cis_values = []
        sas_values = []
        timestep_indices = []

        for i, t in enumerate(timesteps):
            noise_cond, noise_uncond = predict_noise(pipe, latents, t, text_emb, uncond_emb)
            cis = (noise_cond - noise_uncond).float().norm().item()
            cis_values.append(cis)

            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
            latents, pred_x0 = ddim_step(pipe, latents, t, noise_pred, t_prev)

            if i % 5 == 0 or i == len(timesteps) - 1:
                try:
                    pil_img = predicted_x0_to_image(pipe, pred_x0)
                    sas = compute_clip_similarity(
                        clip_model, clip_preprocess, clip_tokenizer, pil_img, prompt
                    )
                except Exception:
                    sas = 0.0
                sas_values.append(sas)
                timestep_indices.append(i)

        all_cis.append(cis_values)
        all_sas.append((timestep_indices, sas_values))

    return {
        "timesteps": [t.item() for t in timesteps],
        "cis_per_prompt": all_cis,
        "sas_per_prompt": all_sas,
        "prompts": prompts,
    }


def run_selective_conditioning(pipe, clip_model, clip_preprocess, clip_tokenizer,
                               prompts, seed=SEED):
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    n_steps = len(timesteps)
    early_end = int(n_steps * 0.4)
    mid_start = early_end
    mid_end = int(n_steps * 0.7)
    late_start = mid_end

    windows = {
        "full": (0, n_steps),
        "early_only": (0, early_end),
        "mid_only": (mid_start, mid_end),
        "late_only": (late_start, n_steps),
        "none": (-1, -1),
    }

    results = {}
    for pi, prompt in enumerate(prompts[:4]):
        print(f"  [Selective] Prompt {pi+1}/4: {prompt[:50]}...")
        text_emb, uncond_emb = encode_prompt(pipe, prompt)
        prompt_results = {}

        for wname, (ws, we) in windows.items():
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            latents = torch.randn(
                (1, pipe.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
                generator=generator, device=DEVICE, dtype=DTYPE,
            )
            latents = latents * pipe.scheduler.init_noise_sigma

            for i, t in enumerate(timesteps):
                noise_cond, noise_uncond = predict_noise(pipe, latents, t, text_emb, uncond_emb)
                if ws <= i < we:
                    noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_uncond
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
                latents, pred_x0 = ddim_step(pipe, latents, t, noise_pred, t_prev)

            final_img = predicted_x0_to_image(pipe, pred_x0)
            clip_score = compute_clip_similarity(
                clip_model, clip_preprocess, clip_tokenizer, final_img, prompt
            )
            prompt_results[wname] = {"image": final_img, "clip_score": clip_score}
            print(f"    {wname}: CLIP={clip_score:.4f}")

        results[prompt] = prompt_results
    return results


def run_trajectory_sensitivity(pipe, prompt_pairs, seed=SEED):
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    all_ts = []

    for pair_idx, (prompt_a, prompt_b) in enumerate(prompt_pairs):
        print(f"  [TS] Pair {pair_idx+1}/{len(prompt_pairs)}")
        text_emb_a, uncond_emb_a = encode_prompt(pipe, prompt_a)
        text_emb_b, uncond_emb_b = encode_prompt(pipe, prompt_b)

        generator_a = torch.Generator(device=DEVICE).manual_seed(seed)
        latents_a = torch.randn(
            (1, pipe.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator_a, device=DEVICE, dtype=DTYPE,
        )
        latents_a = latents_a * pipe.scheduler.init_noise_sigma

        generator_b = torch.Generator(device=DEVICE).manual_seed(seed)
        latents_b = torch.randn(
            (1, pipe.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator_b, device=DEVICE, dtype=DTYPE,
        )
        latents_b = latents_b * pipe.scheduler.init_noise_sigma

        ts_values = []
        for i, t in enumerate(timesteps):
            noise_cond_a, noise_uncond_a = predict_noise(pipe, latents_a, t, text_emb_a, uncond_emb_a)
            noise_pred_a = noise_uncond_a + GUIDANCE_SCALE * (noise_cond_a - noise_uncond_a)

            noise_cond_b, noise_uncond_b = predict_noise(pipe, latents_b, t, text_emb_b, uncond_emb_b)
            noise_pred_b = noise_uncond_b + GUIDANCE_SCALE * (noise_cond_b - noise_uncond_b)

            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
            latents_a, _ = ddim_step(pipe, latents_a, t, noise_pred_a, t_prev)
            latents_b, _ = ddim_step(pipe, latents_b, t, noise_pred_b, t_prev)

            divergence = (latents_a.float() - latents_b.float()).norm().item()
            ts_values.append(divergence)

        all_ts.append(ts_values)

    return {
        "timesteps": [t.item() for t in timesteps],
        "ts_per_pair": all_ts,
        "prompt_pairs": prompt_pairs,
    }


def run_cis_multi_guidance(pipe, prompts, guidance_scales=None, seed=SEED):
    if guidance_scales is None:
        guidance_scales = [3.0, 7.5, 15.0]
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    results = {}

    for gs in guidance_scales:
        print(f"  [CIS-GS] Guidance scale = {gs}")
        all_cis = []
        for pi, prompt in enumerate(prompts[:4]):
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            text_emb, uncond_emb = encode_prompt(pipe, prompt)
            latents = torch.randn(
                (1, pipe.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
                generator=generator, device=DEVICE, dtype=DTYPE,
            )
            latents = latents * pipe.scheduler.init_noise_sigma

            cis_values = []
            for i, t in enumerate(timesteps):
                noise_cond, noise_uncond = predict_noise(pipe, latents, t, text_emb, uncond_emb)
                cis = (noise_cond - noise_uncond).float().norm().item()
                cis_values.append(cis)
                noise_pred = noise_uncond + gs * (noise_cond - noise_uncond)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
                latents, _ = ddim_step(pipe, latents, t, noise_pred, t_prev)
            all_cis.append(cis_values)
        results[gs] = all_cis

    return {
        "timesteps": [t.item() for t in timesteps],
        "cis_per_gs": {str(k): v for k, v in results.items()},
        "guidance_scales": guidance_scales,
    }


# ---- Visualization ----

def plot_cis_curve(data, save_path):
    timesteps = data["timesteps"]
    cis_arr = np.array(data["cis_per_prompt"])
    mean_cis = cis_arr.mean(axis=0)
    std_cis = cis_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(timesteps)), mean_cis, color="#2563eb", linewidth=2, label="Mean CIS")
    ax.fill_between(range(len(timesteps)), mean_cis - std_cis, mean_cis + std_cis,
                     alpha=0.2, color="#2563eb")
    ax.set_xlabel("Denoising Step Index (0 = noisiest)", fontsize=12)
    ax.set_ylabel("Conditioning Influence Score (L2)", fontsize=12)
    ax.set_title("Conditioning Influence Score Across Denoising Steps", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    tick_pos = [0, len(timesteps)//4, len(timesteps)//2, 3*len(timesteps)//4, len(timesteps)-1]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([f"t={timesteps[i]}" for i in tick_pos], fontsize=9)
    ax2.set_xlabel("Diffusion Timestep", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_sas_curve(data, save_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pi, (indices, sas_vals) in enumerate(data["sas_per_prompt"]):
        ax.plot(indices, sas_vals, marker='o', markersize=3, alpha=0.5,
                label=data["prompts"][pi][:30] + "...")
    all_indices = data["sas_per_prompt"][0][0]
    all_sas = np.array([s for _, s in data["sas_per_prompt"]])
    mean_sas = all_sas.mean(axis=0)
    ax.plot(all_indices, mean_sas, color="black", linewidth=2.5,
            marker='s', markersize=5, label="Mean", zorder=10)
    ax.set_xlabel("Denoising Step Index", fontsize=12)
    ax.set_ylabel("CLIP Similarity (Image-Text)", fontsize=12)
    ax.set_title("Semantic Alignment Score Across Denoising Steps", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_selective_conditioning(results, save_path):
    prompts_list = list(results.keys())
    windows = ["full", "early_only", "mid_only", "late_only", "none"]
    wlabels = ["Full Cond.", "Early Only\n(steps 0-40%)",
               "Mid Only\n(steps 40-70%)", "Late Only\n(steps 70-100%)", "No Cond."]

    fig, axes = plt.subplots(len(prompts_list), len(windows),
                              figsize=(3.2 * len(windows), 3.5 * len(prompts_list)))
    for row, prompt in enumerate(prompts_list):
        for col, (wname, wlabel) in enumerate(zip(windows, wlabels)):
            ax = axes[row, col] if len(prompts_list) > 1 else axes[col]
            img = results[prompt][wname]["image"]
            clip_s = results[prompt][wname]["clip_score"]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{wlabel}\nCLIP: {clip_s:.3f}", fontsize=9)
            else:
                ax.set_title(f"CLIP: {clip_s:.3f}", fontsize=9)
        if len(prompts_list) > 1:
            axes[row, 0].set_ylabel(prompt[:25] + "...", fontsize=8, rotation=90, labelpad=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_selective_clip_bar(results, save_path):
    prompts_list = list(results.keys())
    windows = ["full", "early_only", "mid_only", "late_only", "none"]
    wlabels = ["Full", "Early", "Mid", "Late", "None"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(wlabels))
    width = 0.18
    for i, prompt in enumerate(prompts_list):
        scores = [results[prompt][w]["clip_score"] for w in windows]
        ax.bar(x + i * width, scores, width, label=prompt[:25] + "...", alpha=0.85)
    ax.set_xlabel("Conditioning Window", fontsize=12)
    ax.set_ylabel("CLIP Score", fontsize=12)
    ax.set_title("CLIP Scores Under Selective Conditioning", fontsize=13)
    ax.set_xticks(x + width * (len(prompts_list) - 1) / 2)
    ax.set_xticklabels(wlabels)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectory_sensitivity(data, save_path):
    timesteps = data["timesteps"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]
    for i, ts_vals in enumerate(data["ts_per_pair"]):
        pa, pb = data["prompt_pairs"][i]
        label = f"'{pa[:20]}...' vs '{pb[:20]}...'"
        ax.plot(range(len(timesteps)), ts_vals, linewidth=1.8,
                color=colors[i % len(colors)], label=label)
    arr = np.array(data["ts_per_pair"])
    mean_ts = arr.mean(axis=0)
    ax.plot(range(len(timesteps)), mean_ts, color="black", linewidth=2.5,
            linestyle="--", label="Mean", zorder=10)
    ax.set_xlabel("Denoising Step Index", fontsize=12)
    ax.set_ylabel("Trajectory Divergence (L2)", fontsize=12)
    ax.set_title("Trajectory Sensitivity to Conditioning Perturbation", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectory_sensitivity_per_step(data, save_path):
    """Per-step marginal divergence Delta_TS(t) = TS(t) - TS(t-1). Shows where divergence is *added* fastest (slope of cumulative curve)."""
    arr = np.array(data["ts_per_pair"])
    # TS(t) at step index i is the divergence after the i-th denoising step (latents at step i)
    # Per-step gain: delta[i] = TS[i] - TS[i-1], for i >= 1
    delta_ts = np.zeros_like(arr)
    delta_ts[:, 1:] = np.diff(arr, axis=1)
    delta_ts[:, 0] = arr[:, 0]  # first step: gain = TS[0] - 0 = TS[0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]
    for i, d_vals in enumerate(delta_ts):
        pa, pb = data["prompt_pairs"][i]
        label = f"'{pa[:20]}...' vs '{pb[:20]}...'"
        ax.plot(range(len(d_vals)), d_vals, linewidth=1.8,
                color=colors[i % len(colors)], label=label, alpha=0.8)
    mean_delta = delta_ts.mean(axis=0)
    ax.plot(range(len(mean_delta)), mean_delta, color="black", linewidth=2.5,
            linestyle="--", label="Mean", zorder=10)
    ax.set_xlabel("Denoising Step Index", fontsize=12)
    ax.set_ylabel("Per-Step Divergence (L2)", fontsize=12)
    ax.set_title("Per-Step Trajectory Divergence (slope of cumulative TS)", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cis_multi_guidance(data, save_path):
    timesteps = data["timesteps"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2a9d8f", "#2563eb", "#e63946"]
    for ci, gs in enumerate(data["guidance_scales"]):
        cis_arr = np.array(data["cis_per_gs"][str(gs)])
        mean_cis = cis_arr.mean(axis=0)
        std_cis = cis_arr.std(axis=0)
        ax.plot(range(len(timesteps)), mean_cis, linewidth=2, color=colors[ci], label=f"w = {gs}")
        ax.fill_between(range(len(timesteps)), mean_cis - std_cis, mean_cis + std_cis,
                         alpha=0.15, color=colors[ci])
    ax.set_xlabel("Denoising Step Index", fontsize=12)
    ax.set_ylabel("CIS (L2)", fontsize=12)
    ax.set_title("CIS Under Different Guidance Scales", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("Loading Stable Diffusion 1.5 pipeline...")
    pipe = load_sd_pipeline()

    print("Loading CLIP model...")
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model()

    print("\n" + "=" * 60)
    print("Experiment 1 & 4: CIS and Semantic Alignment Score")
    print("=" * 60)
    cis_sas_data = run_cis_and_sas(pipe, clip_model, clip_preprocess, clip_tokenizer, PROMPTS)

    sas_serializable = []
    for indices, vals in cis_sas_data["sas_per_prompt"]:
        sas_serializable.append({"indices": indices, "values": vals})
    save_data = {
        "timesteps": cis_sas_data["timesteps"],
        "cis_per_prompt": cis_sas_data["cis_per_prompt"],
        "sas_per_prompt": sas_serializable,
        "prompts": cis_sas_data["prompts"],
    }
    with open(os.path.join(OUTPUT_DIR, "cis_sas_data.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    plot_cis_curve(cis_sas_data, os.path.join(FIGURE_DIR, "cis_curve.pdf"))
    plot_sas_curve(cis_sas_data, os.path.join(FIGURE_DIR, "sas_curve.pdf"))

    print("\n" + "=" * 60)
    print("Experiment 2: Selective Conditioning Intervention")
    print("=" * 60)
    selective_data = run_selective_conditioning(
        pipe, clip_model, clip_preprocess, clip_tokenizer, PROMPTS
    )

    clip_scores_save = {}
    for prompt, wdata in selective_data.items():
        clip_scores_save[prompt] = {w: d["clip_score"] for w, d in wdata.items()}
    with open(os.path.join(OUTPUT_DIR, "selective_clip_scores.json"), "w") as f:
        json.dump(clip_scores_save, f, indent=2)

    plot_selective_conditioning(selective_data, os.path.join(FIGURE_DIR, "selective_grid.pdf"))
    plot_selective_clip_bar(selective_data, os.path.join(FIGURE_DIR, "selective_clip_bar.pdf"))

    print("\n" + "=" * 60)
    print("Experiment 3: Trajectory Sensitivity")
    print("=" * 60)
    ts_data = run_trajectory_sensitivity(pipe, PROMPT_PAIRS)

    with open(os.path.join(OUTPUT_DIR, "trajectory_sensitivity.json"), "w") as f:
        json.dump(ts_data, f, indent=2)
    plot_trajectory_sensitivity(ts_data, os.path.join(FIGURE_DIR, "trajectory_sensitivity.pdf"))
    plot_trajectory_sensitivity_per_step(ts_data, os.path.join(FIGURE_DIR, "trajectory_sensitivity_per_step.pdf"))

    print("\n" + "=" * 60)
    print("Experiment 5: CIS with Multiple Guidance Scales")
    print("=" * 60)
    gs_data = run_cis_multi_guidance(pipe, PROMPTS)

    with open(os.path.join(OUTPUT_DIR, "cis_multi_guidance.json"), "w") as f:
        json.dump(gs_data, f, indent=2)
    plot_cis_multi_guidance(gs_data, os.path.join(FIGURE_DIR, "cis_multi_guidance.pdf"))

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
