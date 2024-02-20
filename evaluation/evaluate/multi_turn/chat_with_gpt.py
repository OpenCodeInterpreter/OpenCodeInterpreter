# This is an example code. Please implement gpt_predict yourself.

from gradio_client import Client
import json

# Please replace with your Gradio server URL. For Gradio server, please refer to https://huggingface.co/spaces/KikiQiQi/Mediator
URL = "https://your-gradio-server-url.live"

def gpt_predict(messages, model="gpt3.5"):
    inputs = json.dumps(messages)
    client = Client(URL)
    if model == "gpt3.5":
        # Replace with your GPT-3.5 credentials
        args = """{"api_key":"YOUR_GPT3.5_API_KEY","api_type":"azure","model":"gpt-35-turbo","temperature":0,"max_tokens":1024}"""
    elif model == "gpt4":
        # Replace with your GPT-4 credentials
        args = """{"api_key":"YOUR_GPT4_API_KEY","base_url":"https://api.openai.com/v1/","model":"gpt-4-1106-preview","temperature":0,"max_tokens":1024}"""
    result = client.predict(
        inputs,
        args,
        api_name="/submit"
    )
    return result
