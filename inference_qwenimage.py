import torch
from regional_qwenimage_pipeline import RegionalQwenImagePipeline
from diffusers import QwenImagePipeline
import os
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

if __name__ == "__main__":
    pipeline = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16).to("cuda")

    image_width = 1280
    image_height = 1768
    num_inference_steps = 40
    guidance_scale = 3.5
    seed = 124
    base_prompt = "A photo in a modern kitchen: on the left side, a man in a white apron and jeans is chopping vegetables while looking at the camera; on the right side, a woman in a red dress is holding a tray of cookies and smiling at the camera. High quality, 8k, realistic."
    negative_prompt = " "
    import time
    for i in range (1):
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