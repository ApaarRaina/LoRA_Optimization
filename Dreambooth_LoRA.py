import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR      = "Dreambooth_dataset"
PROMPT_FILE      = os.path.join(DATASET_DIR, "prompts.txt")
CLASS_IMAGES_DIR = os.path.join(DATASET_DIR, "class_images")
OUTPUT_DIR       = "trained"


CLASS_PROMPT      = "an business indian man,wearing a suit or casual wear,detailed, sharp focus, photorealistic, colored"
NEGATIVE_PROMPT   = "blurry, low quality, cartoon, painting, illustration, ugly, deformed, watermark, text"
NUM_CLASS_IMAGES  = 100
PRIOR_LOSS_WEIGHT = 1.0

MAX_STEPS  = 1000
SAVE_EVERY = 100
BATCH_SIZE = 2

model_id = "runwayml/stable-diffusion-v1-5"

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def generate_class_images():
    os.makedirs(CLASS_IMAGES_DIR, exist_ok=True)
    existing = len([f for f in os.listdir(CLASS_IMAGES_DIR) if f.endswith(".jpg")])

    if existing >= NUM_CLASS_IMAGES:
        print(f"Class images already exist ({existing}), skipping generation.")
        return

    print(f"Generating {NUM_CLASS_IMAGES} class prior images...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    for i in tqdm(range(existing, NUM_CLASS_IMAGES)):
        image = pipe(
            CLASS_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        image.save(os.path.join(CLASS_IMAGES_DIR, f"class_{i:04d}.jpg"))

    del pipe
    torch.cuda.empty_cache()
    print("Class image generation complete.")


class InstanceDataset(Dataset):
    def __init__(self, prompt_file, transform, split="train"):
        self.transform = transform
        self.data = []
        with open(prompt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, caption = line.split("|", 1)
                if f"{split}/" in path:
                    self.data.append((path.strip(), caption.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), caption


class ClassDataset(Dataset):
    def __init__(self, class_images_dir, class_prompt, transform):
        self.transform = transform
        self.class_prompt = class_prompt
        self.images = [
            os.path.join(class_images_dir, f)
            for f in sorted(os.listdir(class_images_dir))
            if f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        return self.transform(image), self.class_prompt


generate_class_images()

instance_dataset = InstanceDataset(PROMPT_FILE, transform, split="train")
class_dataset    = ClassDataset(CLASS_IMAGES_DIR, CLASS_PROMPT, transform)

instance_loader  = DataLoader(instance_dataset, batch_size=BATCH_SIZE, shuffle=True)
class_loader     = DataLoader(class_dataset,    batch_size=BATCH_SIZE, shuffle=True)

print(f"Instance images: {len(instance_dataset)}")
print(f"Class images:    {len(class_dataset)}")

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

unet         = pipe.unet
vae          = pipe.vae
text_encoder = pipe.text_encoder
tokenizer    = pipe.tokenizer

# VAE frozen — purely a codec, not involved in identity learning
vae.requires_grad_(False)

# UNet LoRA
unet.requires_grad_(False)
unet = get_peft_model(unet, LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none"
))

# Text encoder LoRA — learns what "aadish" means
text_encoder.requires_grad_(False)
text_encoder = get_peft_model(text_encoder, LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    bias="none"
))

print("UNet:"); unet.print_trainable_parameters()
print("Text encoder:"); text_encoder.print_trainable_parameters()

optimizer = torch.optim.AdamW(
    list(unet.parameters()) + list(text_encoder.parameters()),
    lr=5e-6
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_text(captions):
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    return text_encoder(**text_inputs).last_hidden_state


def diffusion_loss(images, captions):
    latents = vae.encode(images).latent_dist.sample() * 0.18215
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, pipe.scheduler.config.num_train_timesteps,
        (latents.shape[0],), device=device
    ).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
    encoder_hidden_states = encode_text(captions)
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    return F.mse_loss(noise_pred, noise)


num_epochs = 1000
step = 0
class_iter = iter(class_loader)

for epoch in range(num_epochs):
    total_loss = 0.0

    for instance_images, instance_captions in tqdm(instance_loader):
        try:
            class_images, class_captions = next(class_iter)
        except StopIteration:
            class_iter = iter(class_loader)
            class_images, class_captions = next(class_iter)

        instance_images = instance_images.to(device)
        class_images    = class_images.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_instance = diffusion_loss(instance_images, list(instance_captions))
            loss_prior    = diffusion_loss(class_images, list(class_captions))
            loss = loss_instance + PRIOR_LOSS_WEIGHT * loss_prior

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % SAVE_EVERY == 0:
            unet.save_pretrained(f"{OUTPUT_DIR}/checkpoint_{step}")
            text_encoder.save_pretrained(f"{OUTPUT_DIR}/checkpoint_{step}_text_encoder")
            print(f"  Saved checkpoint at step {step}")

        if step >= MAX_STEPS:
            break

    avg_loss = total_loss / len(instance_loader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

    if step >= MAX_STEPS:
        print(f"Reached max steps ({MAX_STEPS}), stopping.")
        break

unet.save_pretrained(f"{OUTPUT_DIR}/lora_weights_dummy")
text_encoder.save_pretrained(f"{OUTPUT_DIR}/lora_text_encoder_dummy")
print("Training complete. Final weights saved.")