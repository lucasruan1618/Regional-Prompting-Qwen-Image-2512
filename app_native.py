import gradio as gr
import torch
import os
import time
from diffusers import QwenImagePipeline

# Global variable to hold the pipeline
pipeline = None

def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    print("Loading Native Qwen-Image model...")
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
    
    # Load the native pipeline directly from diffusers
    pipeline = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Model loaded successfully.")
    return pipeline

try:
    load_pipeline()
except Exception as e:
    print(f"Error loading model: {e}")

def generate(
    prompt, neg_prompt, width, height, steps, guidance, seed
):
    global pipeline
    if pipeline is None:
        try:
            load_pipeline()
        except Exception as e:
            return None, f"Error: {str(e)}"

    # Ensure dimensions are divisible by VAE scale factor (typically 16 for this architecture)
    vae_scale = 16
    width = (int(width) // vae_scale) * vae_scale
    height = (int(height) // vae_scale) * vae_scale

    print(f"Generating with: {width}x{height}, Steps: {steps}, CFG: {guidance}, Seed: {seed}")

    start = time.time()
    try:
        image = pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            true_cfg_scale=guidance,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            generator=torch.Generator("cuda").manual_seed(int(seed)),
        ).images[0]
        
        return image, f"Generated in {time.time()-start:.2f}s"
    except Exception as e:
        return None, f"Generation Error: {str(e)}"

with gr.Blocks(title="Qwen-Image-2512 Native Demo") as demo:
    gr.Markdown("# Native Qwen-Image-2512 Demo")
    gr.Markdown("Standard text-to-image generation without regional prompting.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt", 
                value="A modern kitchen, high quality, 8k", 
                lines=3,
                placeholder="Enter your prompt here..."
            )
            neg_prompt = gr.Textbox(
                label="Negative Prompt", 
                value="",
                placeholder="Low quality, blurry..."
            )
            
            with gr.Row():
                steps = gr.Slider(1, 100, 35, step=1, label="Inference Steps")
                guidance = gr.Slider(1.0, 15.0, 3.5, label="Guidance Scale")
            
            with gr.Row():
                width = gr.Number(label="Width", value=1280, step=16)
                height = gr.Number(label="Height", value=768, step=16)
                seed = gr.Number(label="Seed", value=42, precision=0)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Result", type="pil")
            status = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        generate,
        inputs=[prompt, neg_prompt, width, height, steps, guidance, seed],
        outputs=[output_img, status]
    )

if __name__ == "__main__":
    demo.launch(share=True)
