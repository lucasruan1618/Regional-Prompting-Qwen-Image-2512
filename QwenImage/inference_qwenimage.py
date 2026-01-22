import torch
from regional_qwenimage_pipeline import RegionalQwenImagePipeline
from diffusers import QwenImagePipeline
import os
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

if __name__ == "__main__":
    pipeline = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")

    image_width = 1664
    image_height = 928
    num_inference_steps = 35
    guidance_scale = 3.5
    seed = 124
    base_prompt = "An ancient woman stands solemnly holding a blazing torch, while a fierce battle rages in the background, capturing both strength and tragedy in a historical war scene."
    negative_prompt = " "
    import time
    for i in range (3):
        srt = time.time()
        images = pipeline(
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            width=image_width, height=image_height,
            true_cfg_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images
        print("time taken: ", time.time() - srt)

    images[0].save("output_base.jpg")