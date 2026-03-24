import os
import sys
import torch
import subprocess
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image


CODEFORMER_DIR = os.path.expanduser("~/StyleUAI/LoRA_optim/CodeFormer")

device        = "cuda" if torch.cuda.is_available() else "cpu"
model_id      = "runwayml/stable-diffusion-v1-5"
prompt        = "a moja business man, wearing sunglasses, smiling."
prompt_base   = "a business man, wearing sunglasses, smiling."
negative_prompt = (
    "blurry, low quality, cartoon, painting, illustration, "
    "ugly, deformed, watermark, text"
)

SEED           = 42
NUM_STEPS      = 50
GUIDANCE_SCALE = 7.5
CODEFORMER_W   = 0.5

TEMP_INPUT  = "/tmp/lora_output.png"
TEMP_OUTPUT = "/tmp/codeformer_out"

# ─────────────────────────────────────────────────
# Load pipelines
# ─────────────────────────────────────────────────
print("Loading base SD 1.5 pipeline …")
base_pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to(device)
base_pipe.safety_checker = None

print("Loading LoRA pipeline …")
lora_pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to(device)
lora_pipe.safety_checker = None
lora_pipe.unet         = PeftModel.from_pretrained(lora_pipe.unet,         "trained/lora_weights").to(device)
lora_pipe.text_encoder = PeftModel.from_pretrained(lora_pipe.text_encoder, "trained/lora_text_encoder").to(device)


print("Generating base image …")
generator  = torch.manual_seed(SEED)
base_image = base_pipe(
    prompt_base,
    negative_prompt     = negative_prompt,
    num_inference_steps = NUM_STEPS,
    guidance_scale      = GUIDANCE_SCALE,
    generator           = generator,
).images[0]

print("Generating LoRA image …")
generator  = torch.manual_seed(SEED)
lora_image = lora_pipe(
    prompt,
    negative_prompt     = negative_prompt,
    num_inference_steps = NUM_STEPS,
    guidance_scale      = GUIDANCE_SCALE,
    generator           = generator,
).images[0]


print("Restoring LoRA image with CodeFormer …")


lora_image.save(TEMP_INPUT)


subprocess.run([
    sys.executable,
    os.path.join(CODEFORMER_DIR, "inference_codeformer.py"),
    "-i",        TEMP_INPUT,
    "-o",        TEMP_OUTPUT,
    "-w",        str(CODEFORMER_W),
    "--upscale", "1",
], cwd=CODEFORMER_DIR, check=True)


restored_path = os.path.join(TEMP_OUTPUT, "final_results", "lora_output.png")
if not os.path.exists(restored_path):
    final_dir = os.path.join(TEMP_OUTPUT, "final_results")
    files = [f for f in os.listdir(final_dir) if f.endswith(".png")]
    restored_path = os.path.join(final_dir, files[0])

restored_image = Image.open(restored_path).convert("RGB")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(base_image)
axes[0].set_title("Base SD 1.5", fontsize=13)
axes[0].axis("off")

axes[1].imshow(lora_image)
axes[1].set_title("SD 1.5 + LoRA", fontsize=13)
axes[1].axis("off")

axes[2].imshow(restored_image)
axes[2].set_title(f"LoRA + CodeFormer (w={CODEFORMER_W})", fontsize=13)
axes[2].axis("off")

plt.show()