import numpy as np
from PIL import Image
from share_space.utils import inverse_transform

import torch

from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL

import yaml
from share_space.dataset import get_real_dataloaders
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
config_path = "config.yaml"
config = load_config(config_path)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create dataloaders
print("Creating dataloaders...")
train_loader, val_loader = get_real_dataloaders(
    exp_dir=config["data"]["exp_dir"],
    sim_dir=config["data"]["sim_dir"],
    batch_size=config["data"]["batch_size"],
    num_workers=config["data"]["num_workers"],
    img_size=config["model"]["img_size"],
    train_split=config["data"]["train_split"],
    seed=config["data"]["seed"],
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We do not host the weights of the SD3 VAE -- load it from StabilityAI
sd3_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")

pipeline = DiffusionPipeline.from_pretrained(
    "StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
    vae=sd3_vae,
    custom_pipeline="pohaoc2/PixCell-pipeline-ControlNet-fork",
    trust_remote_code=True,
)

pipeline.to(device);

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
uni_transforms = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
uni_model.eval()
uni_model.to(device);



generated_exp_like_images = []
for batch in train_loader:
    for idx, sim_img in enumerate(batch['simulation']):
        sim_img = sim_img.permute(1, 2, 0).cpu().numpy()
        exp_img = batch['shuffled_exp'][idx].permute(1, 2, 0).cpu().numpy()

        sim_img = inverse_transform(sim_img)
        exp_img = inverse_transform(exp_img)

        # exp image (PIL, continuous)
        exp_img = Image.fromarray(
            (exp_img * 255).clip(0, 255).astype(np.uint8)
        )
        exp_img = exp_img.resize((256, 256), resample=Image.BILINEAR)

        # sim mask (binary)
        sim_img = Image.fromarray(sim_img.astype(np.uint8))
        sim_img = sim_img.resize((256, 256), resample=Image.NEAREST)
        sim_img = np.asarray(sim_img)

        # UNI embedding
        uni_inp = uni_transforms(exp_img).unsqueeze(0)
        with torch.inference_mode():
            uni_emb = uni_model(uni_inp.to(device))

        uni_emb = uni_emb.unsqueeze(1)
        uncond = pipeline.get_unconditional_embedding(uni_emb.shape[0])

        samples = pipeline(
            uni_embeds=uni_emb,
            controlnet_input=sim_img.astype(float) * 255,
            negative_uni_embeds=uncond,
            guidance_scale=3,
            num_images_per_prompt=1
        ).images

        generated_exp_like_images.append(samples[0])