# 🎨 Text-to-Image with Stable Diffusion v1.5

This project demonstrates how to generate high-quality images from text prompts using the **Stable Diffusion v1.5** model from RunwayML, powered by 🤗 Hugging Face's `diffusers` library and accelerated on GPU via PyTorch.

✨ Whether you're experimenting with generative AI, exploring creative prompts, or building an image synthesis app — this minimal pipeline gets you started in minutes!

---

## 🖼️ Example Output

<p align="center">
  <img src="images/unicorn_output.jpg" width="480" alt="Example generated image"/>
</p>

---

## 🧠 Model Used

- `runwayml/stable-diffusion-v1-5`  
  A powerful **text-to-image diffusion model** trained to generate photorealistic images from natural language input.

More about the model: [Hugging Face Model Card](https://huggingface.co/runwayml/stable-diffusion-v1-5)

---

## 📌 Features

- ✅ Simple Colab-compatible implementation
- ✅ GPU-accelerated image generation
- ✅ Supports prompt seeding for reproducibility
- ✅ Adjustable number of inference steps and guidance scale
- ✅ Easily extendable for custom prompts or pipelines

---

## 🚀 Getting Started

Install the required packages (done automatically in Colab):

```bash
pip install diffusers transformers accelerate scipy
````

Load the pipeline and generate your first image:

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "Unicorn through rainbow flowers and cloud, highly detailed, cinematic lighting, 8k"
generator = torch.Generator("cuda").manual_seed(42)

image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]
image.save("unicorn_output.jpg")
```

---

## 🎯 Prompt Used

```text
"Unicorn through rainbow flowers and cloud, highly detailed, cinematic lighting, 8k"
```

You can modify the prompt to generate completely new images!

---

## 💾 Dependencies

* `diffusers`
* `transformers`
* `accelerate`
* `scipy`
* `torch`

