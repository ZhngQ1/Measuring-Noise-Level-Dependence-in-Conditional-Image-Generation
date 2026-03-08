"""
Measuring Noise-Level Dependence in Conditional Image Generation
Main experiment script: custom DDIM sampling with CIS, TS, and SAS metrics.
Uses Stable Diffusion 1.5 via HuggingFace diffusers.
Supports text conditioning and ControlNet-style structural conditioning.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import open_clip

# Optional: ControlNet (requires diffusers with ControlNet support)
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    HAS_CONTROLNET = True
except ImportError:
    HAS_CONTROLNET = False

# Optional: SDXL (and SDXL-ControlNet) support
try:
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline
    HAS_SDXL = True
except ImportError:
    HAS_SDXL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Project-relative paths (override with env vars)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR_BASE = os.environ.get("ECE285_OUTPUT_DIR", os.path.join(_PROJECT_ROOT, "results"))
_FIGURE_DIR_BASE = os.environ.get("ECE285_FIGURE_DIR", os.path.join(_PROJECT_ROOT, "report", "figures"))

# These will be set in main() based on --model to keep SD1.5 vs SDXL runs separated.
OUTPUT_DIR = _OUTPUT_DIR_BASE
FIGURE_DIR = _FIGURE_DIR_BASE
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# SD 1.5 base model: default uses Hugging Face cache (runwayml/stable-diffusion-v1-5).
# Override with env SD15_MODEL_ID if you use a local path, e.g. "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_MODEL_ID = os.environ.get("SD15_MODEL_ID", "runwayml/stable-diffusion-v1-5")
# ControlNet model (Canny edge for SD 1.5); also uses HF cache when already downloaded
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_canny"

# SDXL base model. Requires accepting the license on Hugging Face (see RUN_EXPERIMENTS.md).
SDXL_MODEL_ID = os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
# SDXL ControlNet (Canny)
SDXL_CONTROLNET_MODEL_ID = os.environ.get("SDXL_CONTROLNET_MODEL_ID", "diffusers/controlnet-canny-sdxl-1.0")

SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMG_HEIGHT = 512
IMG_WIDTH = 512

# SDXL tends to be much slower/heavier; use lighter defaults unless overridden by CLI.
SDXL_NUM_INFERENCE_STEPS_DEFAULT = 30
SDXL_EVAL_EVERY_DEFAULT = 10

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

PROMPTS_GEOMETRY = [
    "A photograph of a circle and a rectangle",
    "An abstract painting of geometric shapes",
    "A 3D render of a sphere and a cube",
    "A simple sketch of basic geometry",
    "Geometric shapes on a white background",
    "A circle and a box",
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
    scheduler = DDIMScheduler.from_pretrained(SD15_MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15_MODEL_ID,
        scheduler=scheduler,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_controlnet_pipeline():
    """Load SD 1.5 + ControlNet (Canny). Requires diffusers with ControlNet support."""
    if not HAS_CONTROLNET:
        raise RuntimeError("ControlNet requires: pip install diffusers (with ControlNet support)")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD15_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(SD15_MODEL_ID, subfolder="scheduler")
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sdxl_pipeline():
    if not HAS_SDXL:
        raise RuntimeError("SDXL requires a newer diffusers version with StableDiffusionXLPipeline.")
    scheduler = DDIMScheduler.from_pretrained(SDXL_MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        scheduler=scheduler,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    # SDXL optimizations (do not affect SD1.5):
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        try:
            # Newer diffusers
            pipe.enable_vae_tiling()
        except Exception:
            try:
                # Fallback
                pipe.vae.enable_tiling()
            except Exception:
                pass
    return pipe


def load_sdxl_controlnet_pipeline():
    if not HAS_SDXL:
        raise RuntimeError("SDXL ControlNet requires a newer diffusers version (StableDiffusionXLControlNetPipeline).")
    if not HAS_CONTROLNET:
        raise RuntimeError("ControlNet requires: pip install diffusers (with ControlNet support)")
    controlnet = ControlNetModel.from_pretrained(SDXL_CONTROLNET_MODEL_ID, torch_dtype=DTYPE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(SDXL_MODEL_ID, subfolder="scheduler")
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    # SDXL optimizations (do not affect SD1.5):
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass
    return pipe


def prepare_control_image(pipe_cn, image, height=IMG_HEIGHT, width=IMG_WIDTH):
    """Convert PIL/numpy image to ControlNet cond tensor (B=1, RGB, float in [0,1]).
    IMPORTANT: Do NOT normalize to [-1,1] here; diffusers ControlNet expects [0,1] control maps."""
    if isinstance(image, Image.Image):
        image = np.array(image.resize((width, height)))
    elif not isinstance(image, np.ndarray):
        image = np.array(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    return image


def create_synthetic_control_images(n, size=(IMG_HEIGHT, IMG_WIDTH), seed=SEED):
    """Create n reproducible Canny-style control images (no external files)."""
    if not HAS_CV2:
        raise RuntimeError("Synthetic control images require opencv-python: pip install opencv-python")
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        # Deterministic geometric pattern + noise then Canny
        h, w = size[0], size[1]
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        # Draw a few shapes (reproducible)
        cx, cy = w // 2 + (i * 37) % 100 - 50, h // 2 + (i * 53) % 80 - 40
        cv2.circle(canvas, (cx % w, cy % h), 80 + (i * 7) % 40, 0, 2)
        cv2.rectangle(canvas, (100 + i * 50, 100), (300 + i * 20, 400), 0, 2)
        # Add light noise then blur so Canny gives edges
        noise = (rng.random((h, w)) * 30).astype(np.uint8)
        canvas = np.clip(canvas.astype(np.int32) - noise, 0, 255).astype(np.uint8)
        canvas = cv2.GaussianBlur(canvas, (5, 5), 1.0)
        edges = cv2.Canny(canvas, 50, 150)
        rgb = np.stack([edges] * 3, axis=-1)
        images.append(Image.fromarray(rgb))
    return images


def create_prompt_matched_control_images(pipe_cn, prompts, seed=SEED, steps=20):
    """Create prompt-matched Canny control images without external data.
    We first generate a reference image for each prompt (with ControlNet scale=0),
    then extract Canny edges from that reference image as the ControlNet condition.

    This keeps text prompts identical to text-only experiments while providing
    semantically consistent structural conditions."""
    if not HAS_CV2:
        raise RuntimeError("Prompt-matched control images require opencv-python: pip install opencv-python")

    blank = Image.fromarray(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8), mode="RGB")
    controls = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=DEVICE).manual_seed(seed + i)
        # Generate a reference image with structure scale = 0
        try:
            out = pipe_cn(
                prompt=prompt,
                image=blank,
                controlnet_conditioning_scale=0.0,
                num_inference_steps=steps,
                guidance_scale=GUIDANCE_SCALE,
                height=IMG_HEIGHT,
                width=IMG_WIDTH,
                generator=gen,
            )
        except TypeError:
            out = pipe_cn(
                prompt=prompt,
                control_image=blank,
                controlnet_conditioning_scale=0.0,
                num_inference_steps=steps,
                guidance_scale=GUIDANCE_SCALE,
                height=IMG_HEIGHT,
                width=IMG_WIDTH,
                generator=gen,
            )
        ref = out.images[0].convert("RGB")
        ref_np = np.array(ref).astype(np.uint8)
        ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 1.0)
        edges = cv2.Canny(ref_gray, 50, 150)
        rgb = np.stack([edges] * 3, axis=-1)
        controls.append(Image.fromarray(rgb, mode="RGB"))
    return controls


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


def ddim_step(pipe, latents, t, noise_pred, t_prev, clamp_pred_x0=True, clamp_range=(-1, 1)):
    alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
    if t_prev >= 0:
        alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[t_prev]
    else:
        alpha_prod_t_prev = pipe.scheduler.final_alpha_cumprod

    pred_x0 = (latents - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
    if clamp_pred_x0 and clamp_range is not None:
        pred_x0 = pred_x0.clamp(clamp_range[0], clamp_range[1])
    dir_xt = (1 - alpha_prod_t_prev).sqrt() * noise_pred
    prev_latents = alpha_prod_t_prev.sqrt() * pred_x0 + dir_xt
    return prev_latents, pred_x0


def predicted_x0_to_image(pipe, pred_x0):
    """Decode predicted x0 latent to RGB PIL Image in [0, 255]. Safe for CLIP and Canny."""
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    pred_x0_scaled = (pred_x0.float() / scaling_factor).to(pipe.vae.dtype)
    # Guard against occasional NaN/Inf in intermediate states (more common with SDXL fp16).
    pred_x0_scaled = torch.nan_to_num(pred_x0_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        image = pipe.vae.decode(pred_x0_scaled).sample
    # VAE output is in [-1, 1]; normalize to [0, 1] in float32 for precision
    image = image.cpu().float()
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).numpy()
    # Ensure [0, 255] uint8 RGB for downstream (CLIP, Canny)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    rgb_f = np.clip((image * 255.0), 0, 255)
    rgb = np.nan_to_num(rgb_f, nan=0.0, posinf=255.0, neginf=0.0).round().astype(np.uint8)
    pil = Image.fromarray(rgb[0], mode="RGB")
    return pil


def compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, image, text):
    """Image must be PIL RGB in [0, 255] for correct CLIP encoding."""
    if isinstance(image, Image.Image) and image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    text_tokens = clip_tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        img_features = clip_model.encode_image(img_tensor)
        txt_features = clip_model.encode_text(text_tokens)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        similarity = (img_features @ txt_features.T).item()
    return similarity


def _sdxl_prepare_time_ids(pipe_xl, dtype):
    """Build SDXL size/crop conditioning (time_ids) for a fixed resolution."""
    # For our experiments we use a fixed resolution and no crop.
    original_size = (IMG_HEIGHT, IMG_WIDTH)
    target_size = (IMG_HEIGHT, IMG_WIDTH)
    crops_coords_top_left = (0, 0)
    if pipe_xl.text_encoder_2 is None:
        text_encoder_projection_dim = None
    else:
        text_encoder_projection_dim = pipe_xl.text_encoder_2.config.projection_dim
    # _get_add_time_ids requires text_encoder_projection_dim; infer if missing
    if text_encoder_projection_dim is None:
        # Fallback: will be overridden by actual pooled embeds dim if needed
        text_encoder_projection_dim = 1280
    add_time_ids = pipe_xl._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    return add_time_ids.to(DEVICE)


def encode_prompt_sdxl(pipe_xl, prompt, negative_prompt=""):
    """Return SDXL embeddings for conditional and unconditional passes."""
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe_xl.encode_prompt(
        prompt=prompt,
        device=DEVICE,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    # SDXL also conditions on size/crop via time_ids
    if pipe_xl.text_encoder_2 is None:
        proj_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        proj_dim = pipe_xl.text_encoder_2.config.projection_dim
    add_time_ids = pipe_xl._get_add_time_ids(
        (IMG_HEIGHT, IMG_WIDTH),
        (0, 0),
        (IMG_HEIGHT, IMG_WIDTH),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=proj_dim,
    ).to(DEVICE)
    negative_add_time_ids = add_time_ids
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "add_time_ids": add_time_ids,
        "negative_add_time_ids": negative_add_time_ids,
    }


def predict_noise_sdxl(pipe_xl, latents, t, enc):
    """Predict noise for SDXL, returning (noise_cond, noise_uncond)."""
    latent_model_input = pipe_xl.scheduler.scale_model_input(latents, t)
    with torch.no_grad():
        noise_pred_uncond = pipe_xl.unet(
            latent_model_input,
            t,
            encoder_hidden_states=enc["negative_prompt_embeds"],
            added_cond_kwargs={"text_embeds": enc["negative_pooled_prompt_embeds"], "time_ids": enc["negative_add_time_ids"]},
            return_dict=False,
        )[0]
        noise_pred_cond = pipe_xl.unet(
            latent_model_input,
            t,
            encoder_hidden_states=enc["prompt_embeds"],
            added_cond_kwargs={"text_embeds": enc["pooled_prompt_embeds"], "time_ids": enc["add_time_ids"]},
            return_dict=False,
        )[0]
    return noise_pred_cond, noise_pred_uncond


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


def run_cis_and_sas_sdxl(pipe_xl, clip_model, clip_preprocess, clip_tokenizer, prompts, seed=SEED,
                         num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT,
                         eval_every=SDXL_EVAL_EVERY_DEFAULT):
    """SDXL version of CIS + SAS with the same definitions."""
    pipe_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_xl.scheduler.timesteps
    all_cis = []
    all_sas = []

    for pi, prompt in enumerate(prompts):
        print(f"  [CIS+SAS|SDXL] Prompt {pi+1}/{len(prompts)}: {prompt[:50]}...")
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        enc = encode_prompt_sdxl(pipe_xl, prompt)
        latents = torch.randn(
            (1, pipe_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents = latents * pipe_xl.scheduler.init_noise_sigma

        cis_values = []
        sas_values = []
        timestep_indices = []

        for i, t in enumerate(timesteps):
            noise_cond, noise_uncond = predict_noise_sdxl(pipe_xl, latents, t, enc)
            cis = (noise_cond - noise_uncond).float().norm().item()
            cis_values.append(cis)

            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents, pred_x0 = ddim_step(pipe_xl, latents, t, noise_pred, t_prev, clamp_pred_x0=False)

            if i % eval_every == 0 or i == len(timesteps) - 1:
                try:
                    pil_img = predicted_x0_to_image(pipe_xl, pred_x0)
                    sas = compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, pil_img, prompt)
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


def run_selective_conditioning_sdxl(pipe_xl, clip_model, clip_preprocess, clip_tokenizer, prompts, seed=SEED,
                                    num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT):
    pipe_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_xl.scheduler.timesteps
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
        print(f"  [Selective|SDXL] Prompt {pi+1}/4: {prompt[:50]}...")
        enc = encode_prompt_sdxl(pipe_xl, prompt)
        prompt_results = {}

        for wname, (ws, we) in windows.items():
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            latents = torch.randn(
                (1, pipe_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
                generator=generator, device=DEVICE, dtype=DTYPE,
            )
            latents = latents * pipe_xl.scheduler.init_noise_sigma

            for i, t in enumerate(timesteps):
                noise_cond, noise_uncond = predict_noise_sdxl(pipe_xl, latents, t, enc)
                if ws <= i < we:
                    noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_uncond
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                latents, pred_x0 = ddim_step(pipe_xl, latents, t, noise_pred, t_prev, clamp_pred_x0=False)

            final_img = predicted_x0_to_image(pipe_xl, pred_x0)
            clip_score = compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, final_img, prompt)
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


def run_trajectory_sensitivity_sdxl(pipe_xl, prompt_pairs, seed=SEED,
                                    num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT):
    pipe_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_xl.scheduler.timesteps
    all_ts = []

    for pair_idx, (prompt_a, prompt_b) in enumerate(prompt_pairs):
        print(f"  [TS|SDXL] Pair {pair_idx+1}/{len(prompt_pairs)}")
        enc_a = encode_prompt_sdxl(pipe_xl, prompt_a)
        enc_b = encode_prompt_sdxl(pipe_xl, prompt_b)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents_a = torch.randn(
            (1, pipe_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents_a = latents_a * pipe_xl.scheduler.init_noise_sigma
        latents_b = latents_a.clone()

        ts_values = []
        for i, t in enumerate(timesteps):
            noise_cond_a, noise_uncond_a = predict_noise_sdxl(pipe_xl, latents_a, t, enc_a)
            noise_pred_a = noise_uncond_a + GUIDANCE_SCALE * (noise_cond_a - noise_uncond_a)

            noise_cond_b, noise_uncond_b = predict_noise_sdxl(pipe_xl, latents_b, t, enc_b)
            noise_pred_b = noise_uncond_b + GUIDANCE_SCALE * (noise_cond_b - noise_uncond_b)

            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents_a, _ = ddim_step(pipe_xl, latents_a, t, noise_pred_a, t_prev, clamp_pred_x0=False)
            latents_b, _ = ddim_step(pipe_xl, latents_b, t, noise_pred_b, t_prev, clamp_pred_x0=False)

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


def run_cis_multi_guidance_sdxl(pipe_xl, prompts, guidance_scales=None, seed=SEED,
                                num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT):
    if guidance_scales is None:
        guidance_scales = [3.0, 7.5, 15.0]
    pipe_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_xl.scheduler.timesteps
    results = {}

    for gs in guidance_scales:
        print(f"  [CIS-GS|SDXL] Guidance scale = {gs}")
        all_cis = []
        for pi, prompt in enumerate(prompts[:4]):
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            enc = encode_prompt_sdxl(pipe_xl, prompt)
            latents = torch.randn(
                (1, pipe_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
                generator=generator, device=DEVICE, dtype=DTYPE,
            )
            latents = latents * pipe_xl.scheduler.init_noise_sigma

            cis_values = []
            for i, t in enumerate(timesteps):
                noise_cond, noise_uncond = predict_noise_sdxl(pipe_xl, latents, t, enc)
                cis = (noise_cond - noise_uncond).float().norm().item()
                cis_values.append(cis)
                noise_pred = noise_uncond + gs * (noise_cond - noise_uncond)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                latents, _ = ddim_step(pipe_xl, latents, t, noise_pred, t_prev, clamp_pred_x0=False)
            all_cis.append(cis_values)
        results[gs] = all_cis

    return {
        "timesteps": [t.item() for t in timesteps],
        "cis_per_gs": {str(k): v for k, v in results.items()},
        "guidance_scales": guidance_scales,
    }


# =============================================================================
# ControlNet-style structural conditioning experiments
# =============================================================================

CONTROLNET_STRUCT_SCALE = 1.0  # default conditioning scale when structure is "on"


def _controlnet_noise_pred(pipe_cn, latents, t, prompt_embeds, control_image_tensor, struct_scale):
    """Single noise prediction: text + structural conditioning at scale struct_scale.
    prompt_embeds: (B, seq, dim), control_image_tensor: (B, 3, H, W). Returns (B, 4, h, w)."""
    latent_model_input = pipe_cn.scheduler.scale_model_input(latents, t)
    with torch.no_grad():
        try:
            down_res, mid_res = pipe_cn.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_image_tensor,
                return_dict=False,
            )
        except TypeError:
            down_res, mid_res = pipe_cn.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                image=control_image_tensor,
                return_dict=False,
            )
        down_res = [r * struct_scale for r in down_res]
        mid_res = mid_res * struct_scale
        noise_pred = pipe_cn.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample
    return noise_pred


def _controlnet_cfg_step(pipe_cn, latents, t, t_prev, text_emb, uncond_emb, control_image_tensor, struct_scale):
    """One DDIM step with CFG (text) and structural conditioning at struct_scale.
    Duplicates batch for [uncond, cond] and applies CFG. t_prev = next timestep value, or -1 for last step."""
    prompt_embeds = torch.cat([uncond_emb, text_emb], dim=0)
    control_batch = control_image_tensor.repeat(2, 1, 1, 1)
    latent_input = latents.repeat(2, 1, 1, 1)
    noise_pred_both = _controlnet_noise_pred(
        pipe_cn, latent_input, t, prompt_embeds, control_batch, struct_scale
    )
    noise_uncond, noise_cond = noise_pred_both.chunk(2, dim=0)
    noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
    # DDIM step (single batch)
    alpha_prod_t = pipe_cn.scheduler.alphas_cumprod[t]
    if t_prev is not None and t_prev >= 0:
        alpha_prod_t_prev = pipe_cn.scheduler.alphas_cumprod[t_prev]
    else:
        alpha_prod_t_prev = getattr(
            pipe_cn.scheduler, "final_alpha_cumprod",
            pipe_cn.scheduler.alphas_cumprod[0],
        )
    pred_x0 = (latents - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
    pred_x0 = pred_x0.clamp(-1, 1)
    dir_xt = (1 - alpha_prod_t_prev).sqrt() * noise_pred
    prev_latents = alpha_prod_t_prev.sqrt() * pred_x0 + dir_xt
    return prev_latents, pred_x0


def _controlnet_noise_pred_sdxl(pipe_cn_xl, latents, t, prompt_embeds, pooled_prompt_embeds, add_time_ids, control_image_tensor, struct_scale):
    """SDXL ControlNet noise prediction for a single conditioning (either cond or uncond)."""
    latent_model_input = pipe_cn_xl.scheduler.scale_model_input(latents, t)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
    with torch.no_grad():
        # ControlNet forward
        try:
            down_res, mid_res = pipe_cn_xl.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_image_tensor,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        except TypeError:
            down_res, mid_res = pipe_cn_xl.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                image=control_image_tensor,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        down_res = [r * struct_scale for r in down_res]
        mid_res = mid_res * struct_scale
        noise_pred = pipe_cn_xl.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    return noise_pred


def _controlnet_cfg_step_sdxl(pipe_cn_xl, latents, t, t_prev, enc, control_image_tensor, struct_scale):
    """One DDIM step for SDXL ControlNet with CFG (text) and structural conditioning."""
    noise_uncond = _controlnet_noise_pred_sdxl(
        pipe_cn_xl,
        latents,
        t,
        enc["negative_prompt_embeds"],
        enc["negative_pooled_prompt_embeds"],
        enc["negative_add_time_ids"],
        control_image_tensor,
        struct_scale,
    )
    noise_cond = _controlnet_noise_pred_sdxl(
        pipe_cn_xl,
        latents,
        t,
        enc["prompt_embeds"],
        enc["pooled_prompt_embeds"],
        enc["add_time_ids"],
        control_image_tensor,
        struct_scale,
    )
    noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
    prev_latents, pred_x0 = ddim_step(pipe_cn_xl, latents, t, noise_pred, t_prev, clamp_pred_x0=False)
    return prev_latents, pred_x0


def run_cis_struct(pipe_cn, clip_model, clip_preprocess, clip_tokenizer, prompts, control_images, seed=SEED):
    """CIS for structural conditioning: at each step, magnitude of (noise with struct - noise without struct)."""
    pipe_cn.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe_cn.scheduler.timesteps
    n_prompts = min(len(prompts), len(control_images))
    all_cis = []

    for pi in range(n_prompts):
        prompt = prompts[pi]
        ctrl_img = control_images[pi]
        print(f"  [CIS_struct] Prompt {pi+1}/{n_prompts}: {prompt[:40]}...")
        text_emb, uncond_emb = encode_prompt(pipe_cn, prompt)
        ctrl_tensor = prepare_control_image(pipe_cn, ctrl_img)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents = torch.randn(
            (1, pipe_cn.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents = latents * pipe_cn.scheduler.init_noise_sigma

        cis_values = []
        for i, t in enumerate(timesteps):
            # _controlnet_noise_pred internally applies scheduler.scale_model_input(...)
            noise_with = _controlnet_noise_pred(pipe_cn, latents, t, text_emb, ctrl_tensor, 1.0)
            noise_without = _controlnet_noise_pred(pipe_cn, latents, t, text_emb, ctrl_tensor, 0.0)
            cis = (noise_with - noise_without).float().norm().item()
            cis_values.append(cis)
            # Advance latents with full conditioning for next step
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else None
            latents, _ = _controlnet_cfg_step(
                pipe_cn, latents, t, timesteps[i + 1] if i + 1 < len(timesteps) else -1,
                text_emb, uncond_emb, ctrl_tensor, CONTROLNET_STRUCT_SCALE,
            )
        all_cis.append(cis_values)

    return {
        "timesteps": [t.item() for t in timesteps],
        "cis_struct_per_prompt": all_cis,
        "prompts": prompts[:n_prompts],
    }


def run_cis_struct_sdxl(pipe_cn_xl, prompts, control_images, seed=SEED,
                        num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT):
    """SDXL CIS_struct: magnitude of (noise with struct - noise without struct) at each step."""
    pipe_cn_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_cn_xl.scheduler.timesteps
    n_prompts = min(len(prompts), len(control_images))
    all_cis = []

    for pi in range(n_prompts):
        prompt = prompts[pi]
        ctrl_img = control_images[pi]
        print(f"  [CIS_struct|SDXL] Prompt {pi+1}/{n_prompts}: {prompt[:40]}...")
        enc = encode_prompt_sdxl(pipe_cn_xl, prompt)
        ctrl_tensor = prepare_control_image(pipe_cn_xl, ctrl_img)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents = torch.randn(
            (1, pipe_cn_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents = latents * pipe_cn_xl.scheduler.init_noise_sigma

        cis_values = []
        for i, t in enumerate(timesteps):
            noise_with = _controlnet_noise_pred_sdxl(
                pipe_cn_xl, latents, t,
                enc["prompt_embeds"], enc["pooled_prompt_embeds"], enc["add_time_ids"],
                ctrl_tensor, 1.0,
            )
            noise_without = _controlnet_noise_pred_sdxl(
                pipe_cn_xl, latents, t,
                enc["prompt_embeds"], enc["pooled_prompt_embeds"], enc["add_time_ids"],
                ctrl_tensor, 0.0,
            )
            cis_values.append((noise_with - noise_without).float().norm().item())

            # Advance latents with full structure + CFG
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents, _ = _controlnet_cfg_step_sdxl(pipe_cn_xl, latents, t, t_prev, enc, ctrl_tensor, CONTROLNET_STRUCT_SCALE)

        all_cis.append(cis_values)

    return {
        "timesteps": [t.item() for t in timesteps],
        "cis_struct_per_prompt": all_cis,
        "prompts": prompts[:n_prompts],
    }


def run_ts_struct(pipe_cn, prompt, control_image_pairs, seed=SEED):
    """TS for structure: same prompt and noise, two different control images; trajectory L2 divergence."""
    pipe_cn.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe_cn.scheduler.timesteps
    text_emb, uncond_emb = encode_prompt(pipe_cn, prompt)
    all_ts = []

    for pair_idx, (ctrl_a, ctrl_b) in enumerate(control_image_pairs):
        print(f"  [TS_struct] Pair {pair_idx+1}/{len(control_image_pairs)}")
        ctrl_a_t = prepare_control_image(pipe_cn, ctrl_a)
        ctrl_b_t = prepare_control_image(pipe_cn, ctrl_b)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents_a = torch.randn(
            (1, pipe_cn.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents_a = latents_a * pipe_cn.scheduler.init_noise_sigma
        latents_b = latents_a.clone()

        ts_values = []
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents_a, _ = _controlnet_cfg_step(
                pipe_cn, latents_a, t, t_prev, text_emb, uncond_emb, ctrl_a_t, CONTROLNET_STRUCT_SCALE,
            )
            latents_b, _ = _controlnet_cfg_step(
                pipe_cn, latents_b, t, t_prev, text_emb, uncond_emb, ctrl_b_t, CONTROLNET_STRUCT_SCALE,
            )
            divergence = (latents_a.float() - latents_b.float()).norm().item()
            ts_values.append(divergence)
        all_ts.append(ts_values)

    return {
        "timesteps": [t.item() for t in timesteps],
        "ts_struct_per_pair": all_ts,
        "prompt": prompt,
        "num_pairs": len(control_image_pairs),
    }


def run_ts_struct_sdxl(pipe_cn_xl, prompt, control_image_pairs, seed=SEED,
                       num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT):
    """SDXL TS_struct: same prompt + noise, two different control images."""
    pipe_cn_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_cn_xl.scheduler.timesteps
    enc = encode_prompt_sdxl(pipe_cn_xl, prompt)
    all_ts = []

    for pair_idx, (ctrl_a, ctrl_b) in enumerate(control_image_pairs):
        print(f"  [TS_struct|SDXL] Pair {pair_idx+1}/{len(control_image_pairs)}")
        ctrl_a_t = prepare_control_image(pipe_cn_xl, ctrl_a)
        ctrl_b_t = prepare_control_image(pipe_cn_xl, ctrl_b)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents_a = torch.randn(
            (1, pipe_cn_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents_a = latents_a * pipe_cn_xl.scheduler.init_noise_sigma
        latents_b = latents_a.clone()

        ts_values = []
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents_a, _ = _controlnet_cfg_step_sdxl(pipe_cn_xl, latents_a, t, t_prev, enc, ctrl_a_t, CONTROLNET_STRUCT_SCALE)
            latents_b, _ = _controlnet_cfg_step_sdxl(pipe_cn_xl, latents_b, t, t_prev, enc, ctrl_b_t, CONTROLNET_STRUCT_SCALE)
            ts_values.append((latents_a.float() - latents_b.float()).norm().item())

        all_ts.append(ts_values)

    return {
        "timesteps": [t.item() for t in timesteps],
        "ts_struct_per_pair": all_ts,
        "prompt": prompt,
        "num_pairs": len(control_image_pairs),
    }


def _pil_to_uint8_rgb(pil_img, size=None):
    """Ensure we have (H, W) or (H, W, 3) uint8 in [0, 255] for Canny. size = (width, height) for resize."""
    if size is not None:
        pil_img = pil_img.resize(size)
    arr = np.array(pil_img)
    if arr.dtype != np.uint8:
        if arr.size > 0 and np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 and arr.min() >= 0:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return arr


def _structural_similarity_canny(gen_pil, condition_pil):
    """Structural alignment: Canny edges of gen vs condition. Returns IoU in [0,1] (higher = more similar).
    Both inputs must be RGB images in [0, 255] (PIL or already decoded)."""
    if not HAS_CV2:
        return 0.0
    gen = _pil_to_uint8_rgb(gen_pil, size=(IMG_WIDTH, IMG_HEIGHT))
    cond = _pil_to_uint8_rgb(condition_pil, size=(IMG_WIDTH, IMG_HEIGHT))
    if gen.ndim == 3:
        gen = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)
    if cond.ndim == 3:
        cond = cv2.cvtColor(cond, cv2.COLOR_RGB2GRAY)
    e_gen = cv2.Canny(cv2.GaussianBlur(gen, (3, 3), 0.5), 50, 150)
    # If condition is already a binary edge map (e.g., synthetic Canny control), avoid running Canny again.
    cond_vals = np.unique(cond)
    if cond_vals.size <= 4 and np.all(np.isin(cond_vals, [0, 255])):
        e_cond = (cond > 127).astype(np.uint8) * 255
    else:
        e_cond = cv2.Canny(cv2.GaussianBlur(cond, (3, 3), 0.5), 50, 150)
    inter = np.logical_and(e_gen > 0, e_cond > 0).sum()
    union = np.logical_or(e_gen > 0, e_cond > 0).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def run_sas_struct(pipe_cn, clip_model, clip_preprocess, clip_tokenizer, prompts, control_images, seed=SEED):
    """SAS for structure: at sampled steps, decode pred_x0 and compute structural alignment (edge overlap) + CLIP."""
    pipe_cn.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe_cn.scheduler.timesteps
    n_prompts = min(len(prompts), len(control_images))
    sas_struct_all = []
    sas_text_all = []

    for pi in range(n_prompts):
        prompt = prompts[pi]
        ctrl_img = control_images[pi]
        print(f"  [SAS_struct] Prompt {pi+1}/{n_prompts}: {prompt[:40]}...")
        text_emb, uncond_emb = encode_prompt(pipe_cn, prompt)
        ctrl_tensor = prepare_control_image(pipe_cn, ctrl_img)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents = torch.randn(
            (1, pipe_cn.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents = latents * pipe_cn.scheduler.init_noise_sigma

        struct_vals = []
        text_vals = []
        step_indices = []
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents, pred_x0 = _controlnet_cfg_step(
                pipe_cn, latents, t, t_prev, text_emb, uncond_emb, ctrl_tensor, CONTROLNET_STRUCT_SCALE,
            )
            if i % 5 == 0 or i == len(timesteps) - 1:
                try:
                    pil_img = predicted_x0_to_image(pipe_cn, pred_x0)
                    s_struct = _structural_similarity_canny(pil_img, ctrl_img)
                    s_text = compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, pil_img, prompt)
                except Exception:
                    s_struct = 0.0
                    s_text = 0.0
                struct_vals.append(s_struct)
                text_vals.append(s_text)
                step_indices.append(i)
        sas_struct_all.append((step_indices, struct_vals))
        sas_text_all.append((step_indices, text_vals))

    return {
        "timesteps": [t.item() for t in timesteps],
        "sas_struct_per_prompt": sas_struct_all,
        "sas_text_per_prompt": sas_text_all,
        "prompts": prompts[:n_prompts],
    }


def run_sas_struct_sdxl(pipe_cn_xl, clip_model, clip_preprocess, clip_tokenizer, prompts, control_images, seed=SEED,
                        num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT,
                        eval_every=SDXL_EVAL_EVERY_DEFAULT):
    """SDXL SAS_struct: decode pred_x0 across steps; compute struct IoU + CLIP text similarity."""
    pipe_cn_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_cn_xl.scheduler.timesteps
    n_prompts = min(len(prompts), len(control_images))
    sas_struct_all = []
    sas_text_all = []

    for pi in range(n_prompts):
        prompt = prompts[pi]
        ctrl_img = control_images[pi]
        print(f"  [SAS_struct|SDXL] Prompt {pi+1}/{n_prompts}: {prompt[:40]}...")
        enc = encode_prompt_sdxl(pipe_cn_xl, prompt)
        ctrl_tensor = prepare_control_image(pipe_cn_xl, ctrl_img)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents = torch.randn(
            (1, pipe_cn_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
            generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents = latents * pipe_cn_xl.scheduler.init_noise_sigma

        struct_vals = []
        text_vals = []
        step_indices = []
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents, pred_x0 = _controlnet_cfg_step_sdxl(
                pipe_cn_xl, latents, t, t_prev, enc, ctrl_tensor, CONTROLNET_STRUCT_SCALE
            )
            if i % eval_every == 0 or i == len(timesteps) - 1:
                try:
                    pil_img = predicted_x0_to_image(pipe_cn_xl, pred_x0)
                    s_struct = _structural_similarity_canny(pil_img, ctrl_img)
                    s_text = compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, pil_img, prompt)
                except Exception:
                    s_struct = 0.0
                    s_text = 0.0
                struct_vals.append(s_struct)
                text_vals.append(s_text)
                step_indices.append(i)

        sas_struct_all.append((step_indices, struct_vals))
        sas_text_all.append((step_indices, text_vals))

    return {
        "timesteps": [t.item() for t in timesteps],
        "sas_struct_per_prompt": sas_struct_all,
        "sas_text_per_prompt": sas_text_all,
        "prompts": prompts[:n_prompts],
    }


def run_selective_structural(pipe_cn, clip_model, clip_preprocess, clip_tokenizer, prompts, control_images, seed=SEED):
    """Selective structural conditioning: apply structure only in early / mid / late windows."""
    pipe_cn.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe_cn.scheduler.timesteps
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

    n_prompts = min(len(prompts), len(control_images), 4)
    results = {}
    for pi in range(n_prompts):
        prompt = prompts[pi]
        ctrl_img = control_images[pi]
        print(f"  [Selective struct] Prompt {pi+1}/{n_prompts}: {prompt[:40]}...")
        text_emb, uncond_emb = encode_prompt(pipe_cn, prompt)
        ctrl_tensor = prepare_control_image(pipe_cn, ctrl_img)
        prompt_results = {}

        for wname, (ws, we) in windows.items():
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            latents = torch.randn(
                (1, pipe_cn.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
                generator=generator, device=DEVICE, dtype=DTYPE,
            )
            latents = latents * pipe_cn.scheduler.init_noise_sigma

            for i, t in enumerate(timesteps):
                scale = 1.0 if (ws <= i < we) else 0.0
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                latents, pred_x0 = _controlnet_cfg_step(
                    pipe_cn, latents, t, t_prev, text_emb, uncond_emb, ctrl_tensor, scale,
                )

            final_img = predicted_x0_to_image(pipe_cn, pred_x0)
            clip_score = compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, final_img, prompt)
            struct_score = _structural_similarity_canny(final_img, ctrl_img)
            prompt_results[wname] = {"image": final_img, "clip_score": clip_score, "struct_score": struct_score}
            print(f"    {wname}: CLIP={clip_score:.4f} struct={struct_score:.4f}")

        results[prompt] = prompt_results
    return results


def run_selective_structural_sdxl(pipe_cn_xl, clip_model, clip_preprocess, clip_tokenizer, prompts, control_images, seed=SEED,
                                  num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT):
    """SDXL selective structural conditioning: structure on only in early/mid/late windows."""
    pipe_cn_xl.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
    timesteps = pipe_cn_xl.scheduler.timesteps
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

    n_prompts = min(len(prompts), len(control_images), 4)
    results = {}
    for pi in range(n_prompts):
        prompt = prompts[pi]
        ctrl_img = control_images[pi]
        print(f"  [Selective struct|SDXL] Prompt {pi+1}/{n_prompts}: {prompt[:40]}...")
        enc = encode_prompt_sdxl(pipe_cn_xl, prompt)
        ctrl_tensor = prepare_control_image(pipe_cn_xl, ctrl_img)
        prompt_results = {}

        for wname, (ws, we) in windows.items():
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            latents = torch.randn(
                (1, pipe_cn_xl.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
                generator=generator, device=DEVICE, dtype=DTYPE,
            )
            latents = latents * pipe_cn_xl.scheduler.init_noise_sigma

            for i, t in enumerate(timesteps):
                scale = 1.0 if (ws <= i < we) else 0.0
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                latents, pred_x0 = _controlnet_cfg_step_sdxl(
                    pipe_cn_xl, latents, t, t_prev, enc, ctrl_tensor, scale
                )

            final_img = predicted_x0_to_image(pipe_cn_xl, pred_x0)
            clip_score = compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, final_img, prompt)
            struct_score = _structural_similarity_canny(final_img, ctrl_img)
            prompt_results[wname] = {"image": final_img, "clip_score": clip_score, "struct_score": struct_score}
            print(f"    {wname}: CLIP={clip_score:.4f} struct={struct_score:.4f}")

        results[prompt] = prompt_results
    return results


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


# ---- ControlNet visualization ----

def plot_cis_struct_curve(data, save_path):
    timesteps = data["timesteps"]
    cis_arr = np.array(data["cis_struct_per_prompt"])
    mean_cis = cis_arr.mean(axis=0)
    std_cis = cis_arr.std(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(timesteps)), mean_cis, color="#9333ea", linewidth=2, label="Mean CIS (struct)")
    ax.fill_between(range(len(timesteps)), mean_cis - std_cis, mean_cis + std_cis,
                    alpha=0.2, color="#9333ea")
    ax.set_xlabel("Denoising Step Index (0 = noisiest)", fontsize=12)
    ax.set_ylabel("Structural CIS (L2)", fontsize=12)
    ax.set_title("Structural Conditioning Influence Score Across Steps", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_sas_struct_curves(data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Structural alignment
    ax = axes[0]
    for pi, (indices, vals) in enumerate(data["sas_struct_per_prompt"]):
        ax.plot(indices, vals, marker="o", markersize=3, alpha=0.5,
                label=data["prompts"][pi][:25] + "...")
    all_inds = data["sas_struct_per_prompt"][0][0]
    mean_s = np.array([v for _, v in data["sas_struct_per_prompt"]]).mean(axis=0)
    ax.plot(all_inds, mean_s, color="black", linewidth=2.5, marker="s", markersize=5, label="Mean")
    ax.set_xlabel("Denoising Step Index", fontsize=12)
    ax.set_ylabel("Edge overlap (struct)", fontsize=12)
    ax.set_title("SAS Structural Alignment", fontsize=13)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    # Text alignment
    ax = axes[1]
    for pi, (indices, vals) in enumerate(data["sas_text_per_prompt"]):
        ax.plot(indices, vals, marker="o", markersize=3, alpha=0.5,
                label=data["prompts"][pi][:25] + "...")
    mean_t = np.array([v for _, v in data["sas_text_per_prompt"]]).mean(axis=0)
    ax.plot(all_inds, mean_t, color="black", linewidth=2.5, marker="s", markersize=5, label="Mean")
    ax.set_xlabel("Denoising Step Index", fontsize=12)
    ax.set_ylabel("CLIP similarity (text)", fontsize=12)
    ax.set_title("SAS Text Alignment", fontsize=13)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_ts_struct(data, save_path):
    timesteps = data["timesteps"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]
    for i, ts_vals in enumerate(data["ts_struct_per_pair"]):
        ax.plot(range(len(timesteps)), ts_vals, linewidth=1.8, color=colors[i % len(colors)],
                label=f"Pair {i+1}")
    arr = np.array(data["ts_struct_per_pair"])
    mean_ts = arr.mean(axis=0)
    ax.plot(range(len(timesteps)), mean_ts, color="black", linewidth=2.5, linestyle="--", label="Mean")
    ax.set_xlabel("Denoising Step Index", fontsize=12)
    ax.set_ylabel("Trajectory Divergence (L2)", fontsize=12)
    ax.set_title("Trajectory Sensitivity (Structural Conditioning)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_selective_structural_grid(results, save_path):
    prompts_list = list(results.keys())
    windows = ["full", "early_only", "mid_only", "late_only", "none"]
    wlabels = ["Full", "Early Only", "Mid Only", "Late Only", "None"]
    fig, axes = plt.subplots(len(prompts_list), len(windows),
                             figsize=(3.2 * len(windows), 3.5 * len(prompts_list)))
    for row, prompt in enumerate(prompts_list):
        for col, (wname, wlabel) in enumerate(zip(windows, wlabels)):
            ax = axes[row, col] if len(prompts_list) > 1 else axes[col]
            img = results[prompt][wname]["image"]
            clip_s = results[prompt][wname]["clip_score"]
            struct_s = results[prompt][wname]["struct_score"]
            ax.imshow(img)
            ax.axis("off")
            title = f"{wlabel}\nCLIP:{clip_s:.3f} Struct:{struct_s:.3f}"
            ax.set_title(title, fontsize=8)
        if len(prompts_list) > 1:
            axes[row, 0].set_ylabel(prompt[:20] + "...", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_adaptive_schedule(pipe, clip_model, clip_preprocess, clip_tokenizer, prompts, seed=SEED):
    """Compare uniform CFG against time-dependent guidance schedules informed by CIS findings."""
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    n_steps = len(timesteps)
    early_end = int(n_steps * 0.4)
    mid_end = int(n_steps * 0.7)

    schedules = {
        "uniform_7.5": [GUIDANCE_SCALE] * n_steps,
        "early_heavy": [15.0 if i < early_end else 3.0 for i in range(n_steps)],
        "mid_heavy": [3.0 if i < early_end else (15.0 if i < mid_end else 3.0) for i in range(n_steps)],
        "truncated_70": [GUIDANCE_SCALE if i < mid_end else 1.0 for i in range(n_steps)],
        "linear_decay": [GUIDANCE_SCALE * (1.0 - 0.8 * i / (n_steps - 1)) for i in range(n_steps)],
    }

    results = {}
    for sname, ws_list in schedules.items():
        print(f"  [Adaptive] Schedule: {sname}")
        clip_scores = []
        for pi, prompt in enumerate(prompts[:4]):
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            text_emb, uncond_emb = encode_prompt(pipe, prompt)
            latents = torch.randn(
                (1, pipe.unet.config.in_channels, IMG_HEIGHT // 8, IMG_WIDTH // 8),
                generator=generator, device=DEVICE, dtype=DTYPE,
            )
            latents = latents * pipe.scheduler.init_noise_sigma

            for i, t in enumerate(timesteps):
                noise_cond, noise_uncond = predict_noise(pipe, latents, t, text_emb, uncond_emb)
                w_t = ws_list[i]
                noise_pred = noise_uncond + w_t * (noise_cond - noise_uncond)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
                latents, pred_x0 = ddim_step(pipe, latents, t, noise_pred, t_prev)

            final_img = predicted_x0_to_image(pipe, pred_x0)
            clip_score = compute_clip_similarity(clip_model, clip_preprocess, clip_tokenizer, final_img, prompt)
            clip_scores.append(clip_score)
            print(f"    Prompt {pi+1}: CLIP={clip_score:.4f}")

        mean_unet_calls = n_steps * 2
        if sname == "truncated_70":
            mean_unet_calls = mid_end * 2 + (n_steps - mid_end) * 1
        results[sname] = {
            "clip_scores": clip_scores,
            "mean_clip": sum(clip_scores) / len(clip_scores),
            "schedule": ws_list,
            "unet_calls": mean_unet_calls,
        }
        print(f"    Mean CLIP: {results[sname]['mean_clip']:.4f}  UNet calls: {mean_unet_calls}")

    return results


def plot_adaptive_schedule(results, save_path):
    names = list(results.keys())
    labels = {
        "uniform_7.5": "Uniform\n(w=7.5)",
        "early_heavy": "Early-heavy\n(15/3)",
        "mid_heavy": "Mid-heavy\n(3/15/3)",
        "truncated_70": "Truncated\n(7.5 to 70%)",
        "linear_decay": "Linear\ndecay",
    }
    means = [results[n]["mean_clip"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2563eb", "#e63946", "#2a9d8f", "#e9c46a", "#9333ea"]
    bars = ax.bar([labels.get(n, n) for n in names], means, color=colors[:len(names)], alpha=0.85)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Mean CLIP Score", fontsize=12)
    ax.set_title("Adaptive Guidance Schedule Comparison", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) * 1.15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Noise-level dependence experiments (text + ControlNet).")
    parser.add_argument("--model", type=str, default="sd15", choices=["sd15", "sdxl"],
                        help="Which base model family to evaluate: sd15 (Stable Diffusion 1.5) or sdxl (SDXL base).")
    parser.add_argument("--text-only", action="store_true",
                        help="Run only text-conditioning experiments (skip ControlNet).")
    parser.add_argument("--controlnet-only", action="store_true",
                        help="Run only ControlNet structural conditioning experiments (skip text suite).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed for reproducibility (default: {SEED}).")
    parser.add_argument("--sdxl-steps", type=int, default=SDXL_NUM_INFERENCE_STEPS_DEFAULT,
                        help=f"SDXL only: number of inference steps (default: {SDXL_NUM_INFERENCE_STEPS_DEFAULT}).")
    parser.add_argument("--sdxl-eval-every", type=int, default=SDXL_EVAL_EVERY_DEFAULT,
                        help=f"SDXL only: evaluate/decode every N steps for SAS (default: {SDXL_EVAL_EVERY_DEFAULT}).")
    parser.add_argument("--sdxl-vae-fp32", action="store_true",
                        help="SDXL only: decode with VAE in float32 to reduce NaNs/black images (slower but more stable).")
    parser.add_argument("--adaptive-only", action="store_true",
                        help="Run only the adaptive guidance schedule experiment (SD 1.5).")
    args = parser.parse_args()
    seed = args.seed

    if args.text_only and args.controlnet_only:
        raise SystemExit("Choose at most one of --text-only or --controlnet-only (or neither to run both).")

    # Separate outputs for SDXL vs SD1.5 to avoid overwriting results.
    global OUTPUT_DIR, FIGURE_DIR
    if args.model == "sd15":
        OUTPUT_DIR = _OUTPUT_DIR_BASE
        FIGURE_DIR = _FIGURE_DIR_BASE
    else:
        OUTPUT_DIR = os.path.join(_OUTPUT_DIR_BASE, "sdxl")
        FIGURE_DIR = os.path.join(_FIGURE_DIR_BASE, "sdxl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Fix seeds for full reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 60)
    print(f"Random seed: {seed}")
    print(f"Model: {args.model}")
    print("=" * 60)

    print("Loading CLIP model...")
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model()

    # ---------------- ControlNet-only mode ----------------
    if args.controlnet_only:
        if not HAS_CV2:
            print("ControlNet experiments require opencv-python.")
            sys.exit(1)
        if args.model == "sd15":
            if not HAS_CONTROLNET:
                print("ControlNet experiments require diffusers with ControlNet support.")
                sys.exit(1)
            _run_controlnet_experiments(seed, clip_model, clip_preprocess, clip_tokenizer)
        else:
            if not HAS_SDXL:
                print("SDXL requires a newer diffusers version with StableDiffusionXLPipeline.")
                sys.exit(1)
            if not HAS_CONTROLNET:
                print("SDXL ControlNet experiments require diffusers with ControlNet support.")
                sys.exit(1)
            _run_controlnet_experiments_sdxl(
                seed, clip_model, clip_preprocess, clip_tokenizer,
                num_inference_steps=args.sdxl_steps, eval_every=max(1, args.sdxl_eval_every),
                vae_fp32=args.sdxl_vae_fp32,
            )

        print("\n" + "=" * 60)
        print("ControlNet experiments complete.")
        print(f"Results: {OUTPUT_DIR}")
        print(f"Figures: {FIGURE_DIR}")
        print("=" * 60)
        return

    # ---------------- Adaptive schedule experiment ----------------
    if args.adaptive_only:
        print("Loading Stable Diffusion 1.5 pipeline...")
        pipe = load_sd_pipeline()
        print("\n" + "=" * 60)
        print("Adaptive Guidance Schedule Experiment")
        print("=" * 60)
        adaptive_data = run_adaptive_schedule(pipe, clip_model, clip_preprocess, clip_tokenizer, PROMPTS, seed=seed)
        save_adaptive = {k: {"mean_clip": v["mean_clip"], "clip_scores": v["clip_scores"],
                             "unet_calls": v["unet_calls"]} for k, v in adaptive_data.items()}
        with open(os.path.join(OUTPUT_DIR, "adaptive_schedule.json"), "w") as f:
            json.dump(save_adaptive, f, indent=2)
        plot_adaptive_schedule(adaptive_data, os.path.join(FIGURE_DIR, "adaptive_schedule.pdf"))
        print("\n" + "=" * 60)
        print("Adaptive schedule experiment complete.")
        print(f"Results: {OUTPUT_DIR}")
        print(f"Figures: {FIGURE_DIR}")
        print("=" * 60)
        return

    # ---------------- Text suite (sd15 or sdxl) ----------------
    if args.model == "sd15":
        print("Loading Stable Diffusion 1.5 pipeline...")
        pipe = load_sd_pipeline()
    else:
        print("Loading SDXL pipeline...")
        pipe = load_sdxl_pipeline()
        if args.sdxl_vae_fp32:
            pipe.vae.to(torch.float32)

    run_controlnet_after = not args.text_only

    print("\n" + "=" * 60)
    print("Experiment 1 & 4: CIS and Semantic Alignment Score")
    print("=" * 60)
    if args.model == "sd15":
        cis_sas_data = run_cis_and_sas(pipe, clip_model, clip_preprocess, clip_tokenizer, PROMPTS, seed=seed)
    else:
        cis_sas_data = run_cis_and_sas_sdxl(
            pipe, clip_model, clip_preprocess, clip_tokenizer, PROMPTS, seed=seed,
            num_inference_steps=args.sdxl_steps, eval_every=max(1, args.sdxl_eval_every),
        )

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
    if args.model == "sd15":
        selective_data = run_selective_conditioning(pipe, clip_model, clip_preprocess, clip_tokenizer, PROMPTS, seed=seed)
    else:
        selective_data = run_selective_conditioning_sdxl(
            pipe, clip_model, clip_preprocess, clip_tokenizer, PROMPTS, seed=seed,
            num_inference_steps=args.sdxl_steps,
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
    if args.model == "sd15":
        ts_data = run_trajectory_sensitivity(pipe, PROMPT_PAIRS, seed=seed)
    else:
        ts_data = run_trajectory_sensitivity_sdxl(pipe, PROMPT_PAIRS, seed=seed, num_inference_steps=args.sdxl_steps)

    with open(os.path.join(OUTPUT_DIR, "trajectory_sensitivity.json"), "w") as f:
        json.dump(ts_data, f, indent=2)
    plot_trajectory_sensitivity(ts_data, os.path.join(FIGURE_DIR, "trajectory_sensitivity.pdf"))
    plot_trajectory_sensitivity_per_step(ts_data, os.path.join(FIGURE_DIR, "trajectory_sensitivity_per_step.pdf"))

    print("\n" + "=" * 60)
    print("Experiment 5: CIS with Multiple Guidance Scales")
    print("=" * 60)
    if args.model == "sd15":
        gs_data = run_cis_multi_guidance(pipe, PROMPTS, seed=seed)
    else:
        gs_data = run_cis_multi_guidance_sdxl(pipe, PROMPTS, seed=seed, num_inference_steps=args.sdxl_steps)

    with open(os.path.join(OUTPUT_DIR, "cis_multi_guidance.json"), "w") as f:
        json.dump(gs_data, f, indent=2)
    plot_cis_multi_guidance(gs_data, os.path.join(FIGURE_DIR, "cis_multi_guidance.pdf"))

    # -------- ControlNet structural conditioning experiments --------
    if run_controlnet_after:
        if not HAS_CV2:
            print("\n[Skip] ControlNet experiments (missing opencv-python).")
        elif args.model == "sd15":
            if HAS_CONTROLNET:
                _run_controlnet_experiments(seed, clip_model, clip_preprocess, clip_tokenizer)
            else:
                print("\n[Skip] ControlNet experiments (missing diffusers ControlNet).")
        else:
            if HAS_SDXL and HAS_CONTROLNET:
                _run_controlnet_experiments_sdxl(
                    seed, clip_model, clip_preprocess, clip_tokenizer,
                    num_inference_steps=args.sdxl_steps, eval_every=max(1, args.sdxl_eval_every),
                    vae_fp32=args.sdxl_vae_fp32,
                )
            else:
                print("\n[Skip] SDXL ControlNet experiments (missing SDXL or ControlNet pipeline support).")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print("=" * 60)


def _run_controlnet_experiments(seed, clip_model, clip_preprocess, clip_tokenizer):
    """Run ControlNet structural conditioning: CIS_struct, TS_struct, SAS_struct, selective structural."""
    print("\n" + "=" * 60)
    print("ControlNet: Loading pipeline and creating control images...")
    print("=" * 60)
    pipe_cn = load_controlnet_pipeline()
    n_ctrl = 6
    # Keep prompts identical to text-only experiments (control variable),
    # and build prompt-matched Canny control images.
    prompts_cn = PROMPTS[:n_ctrl]
    control_images = create_prompt_matched_control_images(pipe_cn, prompts_cn, seed=seed)
    # Pairs of control images for TS (same prompt, different structure)
    control_pairs = [(control_images[i], control_images[(i + 1) % n_ctrl]) for i in range(4)]
    prompt_ts = prompts_cn[0]

    print("\n  ControlNet Experiment 1: CIS (structural conditioning influence)")
    cis_struct_data = run_cis_struct(
        pipe_cn, clip_model, clip_preprocess, clip_tokenizer,
        prompts_cn, control_images, seed=seed,
    )
    with open(os.path.join(OUTPUT_DIR, "cis_struct_data.json"), "w") as f:
        json.dump({
            "timesteps": cis_struct_data["timesteps"],
            "cis_struct_per_prompt": cis_struct_data["cis_struct_per_prompt"],
            "prompts": cis_struct_data["prompts"],
        }, f, indent=2)
    plot_cis_struct_curve(cis_struct_data, os.path.join(FIGURE_DIR, "cis_struct_curve.pdf"))

    print("\n  ControlNet Experiment 2: Trajectory Sensitivity (structural)")
    ts_struct_data = run_ts_struct(pipe_cn, prompt_ts, control_pairs, seed=seed)
    with open(os.path.join(OUTPUT_DIR, "ts_struct_data.json"), "w") as f:
        json.dump(ts_struct_data, f, indent=2)
    plot_ts_struct(ts_struct_data, os.path.join(FIGURE_DIR, "ts_struct.pdf"))

    print("\n  ControlNet Experiment 3: SAS (structural + text alignment)")
    sas_struct_data = run_sas_struct(
        pipe_cn, clip_model, clip_preprocess, clip_tokenizer,
        prompts_cn, control_images, seed=seed,
    )
    sas_ser = []
    for (inds, s), (_, t) in zip(sas_struct_data["sas_struct_per_prompt"], sas_struct_data["sas_text_per_prompt"]):
        sas_ser.append({"indices": inds, "struct": s, "text": t})
    with open(os.path.join(OUTPUT_DIR, "sas_struct_data.json"), "w") as f:
        json.dump({"timesteps": sas_struct_data["timesteps"], "sas_per_prompt": sas_ser, "prompts": sas_struct_data["prompts"]}, f, indent=2)
    plot_sas_struct_curves(sas_struct_data, os.path.join(FIGURE_DIR, "sas_struct_curves.pdf"))

    print("\n  ControlNet Experiment 4: Selective structural conditioning")
    selective_struct_data = run_selective_structural(
        pipe_cn, clip_model, clip_preprocess, clip_tokenizer,
        prompts_cn[:4], control_images[:4], seed=seed,
    )
    clip_struct_scores = {}
    for prompt, wdata in selective_struct_data.items():
        clip_struct_scores[prompt] = {w: {"clip_score": d["clip_score"], "struct_score": d["struct_score"]} for w, d in wdata.items()}
    with open(os.path.join(OUTPUT_DIR, "selective_structural_scores.json"), "w") as f:
        json.dump(clip_struct_scores, f, indent=2)
    plot_selective_structural_grid(selective_struct_data, os.path.join(FIGURE_DIR, "selective_structural_grid.pdf"))
    print("  ControlNet experiments done.")


def _run_controlnet_experiments_sdxl(seed, clip_model, clip_preprocess, clip_tokenizer,
                                     num_inference_steps=SDXL_NUM_INFERENCE_STEPS_DEFAULT,
                                     eval_every=SDXL_EVAL_EVERY_DEFAULT,
                                     vae_fp32=False):
    """Run SDXL ControlNet structural conditioning experiments."""
    print("\n" + "=" * 60)
    print("ControlNet|SDXL: Loading pipeline and creating control images...")
    print("=" * 60)
    pipe_cn = load_sdxl_controlnet_pipeline()
    if vae_fp32:
        pipe_cn.vae.to(torch.float32)
    n_ctrl = 6
    prompts_cn = PROMPTS[:n_ctrl]
    # Some diffusers versions decode fp16 latents inside pipeline __call__. When VAE
    # is forced to fp32 (--sdxl-vae-fp32), that internal decode can hit a dtype
    # mismatch. Build prompt-matched control images with VAE in fp16, then restore.
    if vae_fp32:
        pipe_cn.vae.to(DTYPE)
    control_images = create_prompt_matched_control_images(pipe_cn, prompts_cn, seed=seed)
    if vae_fp32:
        pipe_cn.vae.to(torch.float32)
    control_pairs = [(control_images[i], control_images[(i + 1) % n_ctrl]) for i in range(4)]
    prompt_ts = prompts_cn[0]

    print("\n  ControlNet|SDXL Experiment 1: CIS (structural conditioning influence)")
    cis_struct_data = run_cis_struct_sdxl(pipe_cn, prompts_cn, control_images, seed=seed, num_inference_steps=num_inference_steps)
    with open(os.path.join(OUTPUT_DIR, "cis_struct_data.json"), "w") as f:
        json.dump({
            "timesteps": cis_struct_data["timesteps"],
            "cis_struct_per_prompt": cis_struct_data["cis_struct_per_prompt"],
            "prompts": cis_struct_data["prompts"],
        }, f, indent=2)
    plot_cis_struct_curve(cis_struct_data, os.path.join(FIGURE_DIR, "cis_struct_curve.pdf"))

    print("\n  ControlNet|SDXL Experiment 2: Trajectory Sensitivity (structural)")
    ts_struct_data = run_ts_struct_sdxl(pipe_cn, prompt_ts, control_pairs, seed=seed, num_inference_steps=num_inference_steps)
    with open(os.path.join(OUTPUT_DIR, "ts_struct_data.json"), "w") as f:
        json.dump(ts_struct_data, f, indent=2)
    plot_ts_struct(ts_struct_data, os.path.join(FIGURE_DIR, "ts_struct.pdf"))

    print("\n  ControlNet|SDXL Experiment 3: SAS (structural + text alignment)")
    sas_struct_data = run_sas_struct_sdxl(
        pipe_cn, clip_model, clip_preprocess, clip_tokenizer, prompts_cn, control_images, seed=seed,
        num_inference_steps=num_inference_steps, eval_every=max(1, eval_every),
    )
    sas_ser = []
    for (inds, s), (_, t) in zip(sas_struct_data["sas_struct_per_prompt"], sas_struct_data["sas_text_per_prompt"]):
        sas_ser.append({"indices": inds, "struct": s, "text": t})
    with open(os.path.join(OUTPUT_DIR, "sas_struct_data.json"), "w") as f:
        json.dump({"timesteps": sas_struct_data["timesteps"], "sas_per_prompt": sas_ser, "prompts": sas_struct_data["prompts"]}, f, indent=2)
    plot_sas_struct_curves(sas_struct_data, os.path.join(FIGURE_DIR, "sas_struct_curves.pdf"))

    print("\n  ControlNet|SDXL Experiment 4: Selective structural conditioning")
    selective_struct_data = run_selective_structural_sdxl(
        pipe_cn, clip_model, clip_preprocess, clip_tokenizer, prompts_cn[:4], control_images[:4], seed=seed,
        num_inference_steps=num_inference_steps,
    )
    clip_struct_scores = {}
    for prompt, wdata in selective_struct_data.items():
        clip_struct_scores[prompt] = {w: {"clip_score": d["clip_score"], "struct_score": d["struct_score"]} for w, d in wdata.items()}
    with open(os.path.join(OUTPUT_DIR, "selective_structural_scores.json"), "w") as f:
        json.dump(clip_struct_scores, f, indent=2)
    plot_selective_structural_grid(selective_struct_data, os.path.join(FIGURE_DIR, "selective_structural_grid.pdf"))
    print("  ControlNet|SDXL experiments done.")


if __name__ == "__main__":
    main()
