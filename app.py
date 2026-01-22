import gradio as gr
import torch
import os
import time
from regional_qwenimage_pipeline import RegionalQwenImagePipeline
from regional_transformer_qwenimage import RegionalQwenImageTransformer2DModel, RegionalQwenImageAttnProcessor2_0

# Global variable to hold the pipeline
pipeline = None

def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    print("Loading Qwen-Image model...")
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
    
    transformer = RegionalQwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image-2512", 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    pipeline = RegionalQwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512", 
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
    
    print("Model loaded successfully.")
    return pipeline

try:
    load_pipeline()
except Exception as e:
    print(f"Error: {e}")

def parse_coords(coord_str, width, height):
    try:
        coords = [int(x.strip()) for x in coord_str.split(',')]
        if len(coords) == 4:
            return coords
    except:
        pass
    return None

def generate(
    base_prompt, bg_prompt, neg_prompt, width, height, steps, guidance, seed,
    mask_steps, base_ratio,
    r1_en, r1_p, r1_c,
    r2_en, r2_p, r2_c
):
    global pipeline
    if pipeline is None: return None, "Pipeline not loaded"

    vae_scale = 16
    width = (int(width) // vae_scale) * vae_scale
    height = (int(height) // vae_scale) * vae_scale

    regional_prompts = []
    regional_masks = []
    background_mask = torch.ones((height, width))

    for en, p, c in [(r1_en, r1_p, r1_c), (r2_en, r2_p, r2_c)]:
        if en and p and c:
            coords = parse_coords(c, width, height)
            if coords:
                x1, y1, x2, y2 = coords
                mask = torch.zeros((height, width))
                mask[y1:y2, x1:x2] = 1.0
                background_mask -= mask
                regional_prompts.append(p)
                regional_masks.append(mask)

    background_mask = torch.clamp(background_mask, min=0.0)
    if background_mask.sum() > 0:
        regional_prompts.append(bg_prompt)
        regional_masks.append(background_mask)

    attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'inject_blocks_interval': 1,
        'base_ratio': float(base_ratio),
    }

    start = time.time()
    image = pipeline(
        prompt=base_prompt,
        negative_prompt=neg_prompt,
        true_cfg_scale=guidance,
        width=width, height=height,
        mask_inject_steps=int(mask_steps),
        num_inference_steps=int(steps),
        generator=torch.Generator("cuda").manual_seed(int(seed)),
        attention_kwargs=attention_kwargs,
    ).images[0]
    
    return image, f"Generated in {time.time()-start:.2f}s"

with gr.Blocks() as demo:
    gr.Markdown("# Regional Prompting Qwen-Image Demo")
    
    with gr.Row():
        with gr.Column():
            base_prompt = gr.Textbox(label="Global Base Prompt", value="A modern kitchen, high quality, 8k")
            neg_prompt = gr.Textbox(label="Negative Prompt", value="")
            
            with gr.Row():
                steps = gr.Slider(1, 100, 40, step=1, label="Inference Steps")
                base_ratio = gr.Slider(0.0, 1.0, 0.3, label="Base Ratio (Control Strength)")
            
            with gr.Row():
                guidance = gr.Slider(1.0, 15.0, 3.5, label="Guidance Scale")
                mask_steps = gr.Slider(0, 50, 10, step=1, label="Mask Inject Steps")

            with gr.Row():
                width = gr.Number(label="Width", value=1280)
                height = gr.Number(label="Height", value=768)
                seed = gr.Number(label="Seed", value=42)

            bg_prompt = gr.Textbox(label="Background Prompt", value="A photo")
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            with gr.Group():
                gr.Markdown("### Region 1 (Left Half)")
                r1_en = gr.Checkbox(label="Enable Region 1", value=True)
                r1_p = gr.Textbox(label="Region 1 Prompt", value="In a modern kitchen, A man in a white apron and jeans, chopping vegetables, looking at camera and a woman in a red dress, holding a tray of cookies, smiling at camera, high quality, 8k, realistic")
                r1_c = gr.Textbox(label="Coords (x1,y1,x2,y2)", value="0,0,640,768")

            with gr.Group():
                gr.Markdown("### Region 2 (Right Half)")
                r2_en = gr.Checkbox(label="Enable Region 2", value=True)
                r2_p = gr.Textbox(label="Region 2 Prompt", value="A woman in a red dress, holding a tray of cookies, smiling at camera.")
                r2_c = gr.Textbox(label="Coords (x1,y1,x2,y2)", value="640,0,1280,768")

            output_img = gr.Image(label="Result")
            status = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        generate,
        [base_prompt, bg_prompt, neg_prompt, width, height, steps, guidance, seed, mask_steps, base_ratio, r1_en, r1_p, r1_c, r2_en, r2_p, r2_c],
        [output_img, status]
    )

if __name__ == "__main__":
    demo.launch(share=True)