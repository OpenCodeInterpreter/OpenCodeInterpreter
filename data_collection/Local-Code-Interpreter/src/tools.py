import openai
import base64
import os
import io
import time
from PIL import Image
from abc import ABCMeta, abstractmethod


def create_vision_chat_completion(vision_model, base64_image, prompt):
    try:
        response = openai.ChatCompletion.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except:
        return None


def create_image(prompt):
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            response_format="b64_json"
        )
        return response.data[0]['b64_json']
    except:
        return None


def image_to_base64(path):
    try:
        _, suffix = os.path.splitext(path)
        if suffix not in {'.jpg', '.jpeg', '.png', '.webp'}:
            img = Image.open(path)
            img_png = img.convert('RGB')
            img_png.tobytes()
            byte_buffer = io.BytesIO()
            img_png.save(byte_buffer, 'PNG')
            encoded_string = base64.b64encode(byte_buffer.getvalue()).decode('utf-8')
        else:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except:
        return None


def base64_to_image_bytes(image_base64):
    try:
        return base64.b64decode(image_base64)
    except:
        return None


def inquire_image(work_dir, vision_model, path, prompt):
    image_base64 = image_to_base64(f'{work_dir}/{path}')
    hypertext_to_display = None
    if image_base64 is None:
        return "Error: Image transform error", None
    else:
        response = create_vision_chat_completion(vision_model, image_base64, prompt)
        if response is None:
            return "Model response error", None
        else:
            return response, hypertext_to_display


def dalle(unique_id, prompt):
    img_base64 = create_image(prompt)
    text_to_gpt = "Image has been successfully generated and displayed to user."

    if img_base64 is None:
        return "Error: Model response error", None

    img_bytes = base64_to_image_bytes(img_base64)
    if img_bytes is None:
        return "Error: Image transform error", None

    temp_path = f'cache/temp_{unique_id}'
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    path = f'{temp_path}/{hash(time.time())}.png'

    with open(path, 'wb') as f:
        f.write(img_bytes)

    hypertext_to_display = f'<img src=\"file={path}\" width="50%" style=\'max-width:none; max-height:none\'>'
    return text_to_gpt, hypertext_to_display


class Tool(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def get_tool_data(self):
        pass


class ImageInquireTool(Tool):
    def support(self):
        return self.config['model']['GPT-4V']['available']

    def get_tool_data(self):
        return {
            "tool_name": "inquire_image",
            "tool": inquire_image,
            "system_prompt": "If necessary, utilize the 'inquire_image' tool to query an AI model regarding the "
                             "content of images uploaded by users. Avoid phrases like\"based on the analysis\"; "
                             "instead, respond as if you viewed the image by yourself. Keep in mind that not every"
                             "tasks related to images require knowledge of the image content, such as converting "
                             "an image format or extracting image file attributes, which should use `execute_code` "
                             "tool instead. Use the tool only when understanding the image content is necessary.",
            "tool_description": {
                "name": "inquire_image",
                "description": "This function enables you to inquire with an AI model about the contents of an image "
                               "and receive the model's response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path of the image"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The question you want to pose to the AI model about the image"
                        }
                    },
                    "required": ["path", "prompt"]
                }
            },
            "additional_parameters": {
                "work_dir": lambda bot_backend: bot_backend.jupyter_work_dir,
                "vision_model": self.config['model']['GPT-4V']['model_name']
            }
        }


class DALLETool(Tool):
    def support(self):
        return True

    def get_tool_data(self):
        return {
            "tool_name": "dalle",
            "tool": dalle,
            "system_prompt": "If user ask you to generate an art image, you can translate user's requirements into a "
                             "prompt and sending it to the `dalle` tool. Please note that this tool is specifically "
                             "designed for creating art images. For scientific figures, such as plots, please use the "
                             "Python code execution tool `execute_code` instead.",
            "tool_description": {
                "name": "dalle",
                "description": "This function allows you to access OpenAI's DALL·E-3 model for image generation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the image you want to generate, should be in "
                                           "English only. "
                        }
                    },
                    "required": ["prompt"]
                }
            },
            "additional_parameters": {
                "unique_id": lambda bot_backend: bot_backend.unique_id,
            }
        }


def get_available_tools(config):
    tools = [ImageInquireTool]

    available_tools = []
    for tool in tools:
        tool_instance = tool(config)
        if tool_instance.support():
            available_tools.append(tool_instance.get_tool_data())
    return available_tools
