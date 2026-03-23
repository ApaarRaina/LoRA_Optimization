import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
prompt = "a moja business man,wearing sunglasses,smiling."
prompt_base = "a business man,wearing sunglasses,smiling."
negative_prompt="blurry, low quality, cartoon, painting, illustration, ugly, deformed, watermark, text"

base_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
base_pipe.safety_checker = None

lora_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
lora_pipe.safety_checker = None

# Load LoRA weights into UNet
lora_pipe.unet = PeftModel.from_pretrained(lora_pipe.unet, "trained/lora_weights").to(device)
lora_pipe.safety_checker = None

# Load LoRA weights into text encoder
lora_pipe.text_encoder = PeftModel.from_pretrained(lora_pipe.text_encoder, "trained/lora_text_encoder").to(device)
lora_pipe.safety_checker = None

seed = 42
generator = torch.manual_seed(seed)

base_image = base_pipe(
    prompt_base,
    negative_prompt=negative_prompt,
    num_inference_steps=100,
    generator=generator
).images[0]

generator = torch.manual_seed(seed)
lora_image = lora_pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=100,
    generator=generator
).images[0]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(base_image)
plt.title("Base SD 1.5")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lora_image)
plt.title("SD 1.5 + LoRA")
plt.axis("off")

plt.tight_layout()
plt.show()