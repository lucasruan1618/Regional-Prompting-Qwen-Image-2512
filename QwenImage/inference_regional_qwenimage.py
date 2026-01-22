import torch
from regional_qwenimage_pipeline import RegionalQwenImagePipeline
from regional_transformer_qwenimage import RegionalQwenImageTransformer2DModel, RegionalQwenImageAttnProcessor2_0
from diffusers import QwenImagePipeline
import os
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

if __name__ == "__main__":
    transformer = RegionalQwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image", 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline = RegionalQwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image", 
        transformer=transformer, 
        torch_dtype=torch.bfloat16
    ).to("cuda")
     
    attn_procs = {}
    for name in pipeline.transformer.attn_processors.keys():
        if 'transformer_blocks' in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalQwenImageAttnProcessor2_0()
        else:
            attn_procs[name] = pipeline.transformer.attn_processors[name]
    pipeline.transformer.set_attn_processor(attn_procs)

    ## generation settings
    
    # example regional prompt and mask pairs
    image_width = 1280
    image_height = 768
    num_samples = 1
    num_inference_steps = 35
    guidance_scale = 3.5
    seed = 124
    base_prompt = "romantic, vibe, lawn, light, broad, high quality, 8k, realistic"
    background_prompt = "romantic, vibe, lawn, light, broad, high quality, 8k, realistic"
    regional_prompt_mask_pairs = {
        "0": {
            "description": "tree, summer, green leaves, high quality, 8k, realistic",
            "mask": [0, 0, image_width//2, image_height]
        },
        "1": {
            "description": "tree, winter, snow, high quality, 8k, realistic",
            "mask": [image_width//2, 0, image_width, image_height]
        }
    }
    # region control settings
    mask_inject_steps = 15
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
            true_cfg_scale = 1.0,
            width=image_width, height=image_height,
            mask_inject_steps=mask_inject_steps,
            # guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            attention_kwargs=attention_kwargs,
        ).images
        print("time taken: ", time.time() - srt)

    images[0].save("output.jpg")