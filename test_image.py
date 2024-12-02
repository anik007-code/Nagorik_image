# to test the model ...
from PIL import Image
from diffusers import StableDiffusionPipeline
from nagorik_imageFT import transform

pipe = StableDiffusionPipeline.from_pretrained('./Model/nagorik_diffusion_model')
pipe.to("cpu")

new_input_image = Image.open("images/sketch3.jpg").convert("RGB")
new_input_image = transform(new_input_image).unsqueeze(0).to("cpu")

generated_image = pipe(new_input_image).images[0]

generated_image.save("generated_image.jpg")
generated_image.show()

