import os
from pathlib import Path
from glob import glob
import shutil
import torch

import gradio as gr
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks, ui
from modules import generation_parameters_copypaste as parameters_copypaste
import launch

from blip2 import BLIP2

script_dir = Path(scripts.basedir())
captioners = {}  

model_list = [
    "coco",
    "pretrain",
]

sampling_methods = ["Nucleus", "Top-K"]

def model_check(name):
    if name not in captioners:
        # library_check()
        if name in model_list:
            print(f"Loading {name} model...")
            # unload other models
            unload_models(False)
            captioners[name] = BLIP2(
                name
            )
            print(f"Model {name} loaded")


def unload_models(log: bool = True):
    if log:
        print("Unloading models...")
    for key in captioners:
        captioners[key].unload()
    captioners.clear()
    torch.cuda.empty_cache()
    if log:
        print("Done. Models unloaded")


def generate_caption(
        image: Image, 
        # model_name, 
        sampling_type,
        num_beams,  
        caption_max_length,
        caption_min_length,
        top_p,
        # repitition_penalty,
    ):
    model_name = "coco"
    model_check(model_name)
    print(f"Generating captions...")
    captions = captioners[model_name].generate_caption(
        image, 
        num_beams=num_beams,
        use_nucleus_sampling=(sampling_type == "Nucleus"),
        max_length=caption_max_length,
        min_length=caption_min_length,
        top_p=top_p,
        # repetition_penalty=repitition_penalty,
    )
    caption = captions[0]
    print(caption)

    return caption
    

def create_caption_file(caption, output_caption_path):
    with open(output_caption_path, "w", encoding="utf-8") as f:
        f.write(caption)

def batch_captioning(
    input_dir, 
    output_dir,
    caption_ext, 
    # model_name,
    sampling_type, 
    num_beams, 
    caption_max_length, 
    caption_min_length, 
    top_p, 
    # repitition_penalty, 
):
    print("Batch captioning started")
    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        image_paths = [
            p
            for p in input_dir.iterdir()
            if (p.is_file and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"])
        ]

        print(f"Found {len(image_paths)} images")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, f in enumerate(image_paths):
            if f.is_dir():
                continue

            img = Image.open(f)
            # fix png
            if img.mode == "RGBA":
                img = img.convert("RGB")

            caption = generate_caption(
                img, 
                # model_name,
                sampling_type=(sampling_type == "Nucleus"),
                num_beams=num_beams,
                caption_max_length=caption_max_length,
                caption_min_length=caption_min_length,
                top_p=top_p,
                # repitition_penalty=repitition_penalty,
            )

            # without ext
            filename = f.stem

            if caption_ext.startswith("."): # remove dot
                caption_ext = caption_ext[1:]

            caption_output_path = output_dir / f"{filename}.{caption_ext}"

            if caption_output_path.exists():
                print(f"File already exists: {caption_output_path}. Skipped.")
                continue

            create_caption_file(caption, caption_output_path)

            print(
                f"{f.name}: {caption}"
            )

        print("All done!")
        return "Done!"
    except Exception as e:
        return f"Error: {e}"


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem(label="Single"):
                    with gr.Row().style(equal_height=False):
                        with gr.Column():
                            image = gr.Image(
                                source="upload",
                                label="Image",
                                interactive=True,
                                type="pil",
                            )

                            # single_model_select = gr.Dropdown(
                            #     label="Model",
                            #     choices=model_list,
                            #     value=model_list[0],
                            #     interactive=True,
                            # )

                            single_sampling_method_radio = gr.Radio(
                                label="Sampling method",
                                choices=sampling_methods,
                                value=sampling_methods[0],
                                interactive=True,
                            )

                            single_number_of_beams_slider = gr.Slider(
                                label="Number of beams (0 = no beam search)",
                                minimum=0,
                                maximum=10,
                                step=1,
                                value=3,
                                interactive=True,
                            )

                            with gr.Row():
                                single_caption_min_length_slider = gr.Slider(
                                    label="Caption min length",
                                    minimum=0,
                                    maximum=200,
                                    step=1,
                                    value=10,
                                    interactive=True,
                                )
                                single_caption_max_length_slider = gr.Slider(
                                    label="Caption max length",
                                    minimum=1,
                                    maximum=200,
                                    step=1,
                                    value=30,
                                    interactive=True,
                                )

                            single_top_p_slider = gr.Slider(
                                label="Top p",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.9,
                                interactive=True,
                            )
                            # single_repetition_penalty_slider = gr.Slider(
                            #     label="Repetition penalty (1.0 = no penalty)",
                            #     minimum=0.0,
                            #     maximum=1.0,
                            #     step=0.01,
                            #     value=1.0,
                            #     interactive=True,
                            # )

                            single_start_btn = gr.Button(
                                value="Interrogate", variant="primary"
                            )

                            single_unload_models_btn = gr.Button(
                                value="Unload models", variant="secondary"
                            )

                        with gr.Column():
                            # single_caption_result = gr.TextArea(
                            #     label="Generated Caption",
                            #     value="",
                            #     readonly=True,
                            #     placeholder="No caption yet",
                            # )
                            single_caption_result = gr.Label(
                                label="Generated Caption",
                            )

                with gr.TabItem(label="Batch"):

                    with gr.Row().style(equal_height=False):
                        with gr.Column():
                            input_dir_input = gr.Textbox(
                                label="Image Directory",
                                placeholder="path/to/caption",
                                type="text",
                            )
                            output_dir_input = gr.Textbox(
                                label="Output Directory",
                                placeholder="path/to/output",
                                type="text",
                            )
                            output_caption_ext = gr.Textbox(
                                label="Output Caption Extension",
                                placeholder="txt",
                                value="txt",
                                type="text",
                            )

                            gr.Markdown("")

                            # batch_model_select = gr.Dropdown(
                            #     label="Model",
                            #     choices=model_list,
                            #     value=model_list[0],
                            #     interactive=True,
                            # )

                            batch_sampling_method_radio = gr.Radio(
                                label="Sampling method",
                                choices=sampling_methods,
                                value=sampling_methods[0],
                                interactive=True,
                            )

                            batch_number_of_beams_slider = gr.Slider(
                                label="Number of beams (0 = no beam search)",
                                minimum=0,
                                maximum=10,
                                step=1,
                                value=3,
                                interactive=True,
                            )

                            with gr.Row():
                                batch_caption_min_length_slider = gr.Slider(
                                    label="Caption min length",
                                    minimum=0,
                                    maximum=200,
                                    step=1,
                                    value=10,
                                    interactive=True,
                                )
                                batch_caption_max_length_slider = gr.Slider(
                                    label="Caption max length",
                                    minimum=1,
                                    maximum=200,
                                    step=1,
                                    value=30,
                                    interactive=True,
                                )
                                

                            batch_top_p_slider = gr.Slider(
                                label="Top p",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.9,
                                interactive=True,
                            )
                            # batch_repetition_penalty_slider = gr.Slider(
                            #     label="Repetition penalty (1.0 = no penalty)",
                            #     minimum=0.0,
                            #     maximum=1.0,
                            #     step=0.01,
                            #     value=1.0,
                            #     interactive=True,
                            # )

                            gr.Markdown("")

                            batch_start_btn = gr.Button(
                                value="Interrogate", variant="primary"
                            )

                            batch_unload_models_btn = gr.Button(
                                value="Unload models", variant="secondary"
                            )

                        with gr.Column():
                            status_block = gr.Label(label="Status", value="Idle")

        image.change(
            fn=generate_caption,
            inputs=[
                image, 
                # single_model_select,
                single_sampling_method_radio, 
                single_number_of_beams_slider, 
                single_caption_max_length_slider, 
                single_caption_min_length_slider, 
                single_top_p_slider, 
                # single_repetition_penalty_slider
            ],
            outputs=single_caption_result,
        )
        single_start_btn.click(
            fn=generate_caption,
            inputs=[
                image, 
                # single_model_select,
                single_sampling_method_radio, 
                single_number_of_beams_slider, 
                single_caption_max_length_slider, 
                single_caption_min_length_slider, 
                single_top_p_slider, 
                # single_repetition_penalty_slider
            ],
            outputs=single_caption_result,
        )
        single_unload_models_btn.click(
            fn=unload_models,
            inputs=[],
            outputs=[],
        )

        batch_start_btn.click(
            fn=batch_captioning,
            inputs=[
                input_dir_input,
                output_dir_input,
                output_caption_ext,
                # batch_model_select,
                batch_sampling_method_radio,
                batch_number_of_beams_slider,
                batch_caption_max_length_slider,
                batch_caption_min_length_slider,
                batch_top_p_slider,
                # batch_repetition_penalty_slider,
            ],
            outputs=[status_block],
        )
        batch_unload_models_btn.click(
            fn=unload_models,
            inputs=[],
            outputs=[],
        )

    return [(ui, "BLIP2 Captioner", "blip2_captioner")]


script_callbacks.on_ui_tabs(on_ui_tabs)
