from diffusers import StableDiffusionPipeline
from diffusers_api.base.base import BaseModel


class StableDiffusionText2ImageModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        """
        Initializes the Stable Diffusion text to image model.

        :param model_name: The name of the pretrained model.
        :param kwargs: Additional parameters for the model.
        """
        super().__init__()
        self.load_model(StableDiffusionPipeline, model_name, **kwargs)

    def generate_image(self, prompt: str, height: int = None, width: int = None, num_inference_steps: int = 50, 
                       guidance_scale: float = 7.5, negative_prompt: str = None, num_images_per_prompt: int = 1, 
                       **kwargs):
        """
        Generates an image based on the given prompt and parameters.

        :param prompt: The prompt to guide the image generation.
        :param height: The height in pixels of the generated image.
        :param width: The width in pixels of the generated image.
        :param num_inference_steps: The number of denoising steps.
        :param guidance_scale: The scale of guidance for the generation.
        :param negative_prompt: The prompt to guide what to not include in image generation.
        :param num_images_per_prompt: The number of images to generate per prompt.
        :param kwargs: Additional parameters for Stable Diffusion's `__call__` method.
        :return: Generated image.
        """
        return self.model(prompt, height=height, width=width, num_inference_steps=num_inference_steps, 
                          guidance_scale=guidance_scale, negative_prompt=negative_prompt, 
                          num_images_per_prompt=num_images_per_prompt, **kwargs)
