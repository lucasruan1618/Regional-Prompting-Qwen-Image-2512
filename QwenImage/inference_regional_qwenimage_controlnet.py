import torch
from regional_qwenimage_controlnet_pipeline import RegionalQwenImageControlNetPipeline
from regional_transformer_qwenimage import RegionalQwenImageTransformer2DModel, RegionalQwenImageAttnProcessor2_0
from diffusers.utils import load_image

from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel
import os
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"



base_model = "Qwen/Qwen-Image"
controlnet_model = "InstantX/Qwen-Image-ControlNet-Union"
controlnet = QwenImageControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
transformer = RegionalQwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image", 
    subfolder="transformer", 
    torch_dtype=torch.bfloat16
).to("cuda")

pipeline = RegionalQwenImageControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, transformer=transformer, torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

attn_procs = {}
for name in pipeline.transformer.attn_processors.keys():
    if 'transformer_blocks' in name and name.endswith("attn.processor"):
        attn_procs[name] = RegionalQwenImageAttnProcessor2_0()
    else:
        attn_procs[name] = pipeline.transformer.attn_processors[name]
pipeline.transformer.set_attn_processor(attn_procs)

# canny
# it is highly suggested to add 'TEXT' into prompt if there are text elements
control_image = load_image("/workspace/research/Regional-Prompting-QwenImage/condition_depth.png")
# example regional prompt and mask pairs
image_width = 1280
image_height = 960
num_samples = 1
num_inference_steps = 50
guidance_scale = 4.0
seed = 124
base_prompt = "an anime image of three high-performance sports cars, red, blue, and yellow, are racing side by side on a city street"
background_prompt = "a photo, 8k, realistic"
regional_prompt_mask_pairs = {
    "0": {
        "description": "An anime sleek blue sports car in the lead position, with aggressive aerodynamic styling and gleaming paint that catches the light. The car appears to be moving at high speed with motion blur effects.",
        "mask": [0, 0, 426, 960]
    },
    "1": {
        "description": "An anime powerful red sports car in the middle position, neck-and-neck with its competitors. Its metallic paint shimmers as it races forward, with visible speed lines and dynamic movement.",
        "mask": [426, 0, 853, 960]
    },
    "2": {
        "description": "An anime striking yellow sports car in the third position, its bold color standing out against the street. The car's aggressive stance and aerodynamic profile emphasize its racing performance.",
        "mask": [853, 0, 1280, 960]
    }
}
# region control settings
mask_inject_steps = 0
inject_blocks_interval = 1
base_ratio = 0.3

## prepare regional prompts and masks
# ensure image width and height are divisible by the vae scale factor
image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor

regional_prompts = []
regional_masks = []
background_mask = torch.ones((image_height, image_width))

for region_idx, region in regional_prompt_mask_pairs.items():
    description = region['description']
    mask = region['mask']
    x1, y1, x2, y2 = mask

    mask = torch.zeros((image_height, image_width))
    mask[y1:y2, x1:x2] = 1.0

    background_mask -= mask

    regional_prompts.append(description)
    regional_masks.append(mask)
        
# if regional masks don't cover the whole image, append background prompt and mask
if background_mask.sum() > 0:
    regional_prompts.append(background_prompt)
    regional_masks.append(background_mask)

# setup regional kwargs that pass to the pipeline
attention_kwargs = {
    'regional_prompts': regional_prompts,
    'regional_masks': regional_masks,
    'inject_blocks_interval': inject_blocks_interval,
    'base_ratio': base_ratio,
}
import time
for i in range (1):
    srt = time.time()
    images = pipeline(
        prompt=base_prompt,
        negative_prompt="ugly, blurry, low quality",
        width=image_width, height=image_height,
        true_cfg_scale = 3.5,
        mask_inject_steps=mask_inject_steps,
        control_image=control_image,
        controlnet_conditioning_scale=1.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        attention_kwargs=attention_kwargs,
    ).images
    print("time taken: ", time.time() - srt)
    images[0].save("output_controlnet.jpg")