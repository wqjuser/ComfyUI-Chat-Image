import base64
from io import BytesIO

import numpy as np
from PIL import Image
from openai import OpenAI


class LLMImageDescription:
    def __init__(self):
        self.output_dir = "output"
        self.type = "output"
        self._client = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-latest", "gemini-2.0-flash-exp"],),
                "api_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True
                }),
                "prompt_template": ("STRING", {
                    "default": "Please describe this image in detail:",
                    "multiline": True
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "process_image"
    CATEGORY = "image/text"

    def get_client(self, api_key, api_url=None):
        """Get or create OpenAI client"""
        if not self._client:
            kwargs = {"api_key": api_key}
            if api_url:
                kwargs["base_url"] = api_url
            self._client = OpenAI(api_key=api_key, base_url=api_url)
        return self._client

    @staticmethod
    def convert_image_to_base64(image):
        # Convert PyTorch tensor to PIL Image
        image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        if image.shape[0] == 3:  # If image is in CHW format
            image = np.transpose(image, (1, 2, 0))
        pil_image = Image.fromarray(image)

        # Convert PIL Image to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    @staticmethod
    def process_with_openai_compatible(prompt_template, base64_image, client, model):
        """Process image using OpenAI API format"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_template
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error: {str(e)}")
            return f"Error: API request failed - {str(e)}"

    def process_image(self, image, model, api_url, api_key, prompt_template):
        try:
            # Convert the first image in batch to base64
            if len(image.shape) == 4:
                image = image[0]
            base64_image = self.convert_image_to_base64(image)

            # Get appropriate API URL based on model
            if not api_url:
                api_urls = {
                    "gpt-4o": "https://api.openai.com/v1/chat/completions",
                    "claude-3-5-sonnet-20241022": "https://api.anthropic.com/v1/messages",
                    "gemini-1.5-pro-latest": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent",
                    "gemini-2.0-flash-exp": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
                }
                api_url = api_urls.get(model, "")

            # Process using unified OpenAI SDK
            print(f'args is api_url: {api_url},api_key: {api_key}')
            client = self.get_client(api_key, api_url)
            description = self.process_with_openai_compatible(prompt_template, base64_image, client, model)

            return (description,)

        except Exception as e:
            print(f"Error in image description: {str(e)}")
            return (f"Error: Failed to generate image description. {str(e)}",)

    def __del__(self):
        """Cleanup client on deletion"""
        if self._client:
            self._client.close()
