import ast
import gradio as gr
import os
import re
import json
import logging

import torch
from datetime import datetime

from threading import Thread
from typing import Optional
from transformers import TextIteratorStreamer
from functools import partial
from huggingface_hub import CommitScheduler
from uuid import uuid4
from pathlib import Path

from code_interpreter.JupyterClient import JupyterNotebook

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from code_interpreter.OpenCodeInterpreter import OpenCodeInterpreter

JSON_DATASET_DIR = Path("json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

scheduler = CommitScheduler(
    repo_id="opencodeinterpreter_user_data",
    repo_type="dataset",
    folder_path=JSON_DATASET_DIR,
    path_in_repo="data",
    private=True
)

logging.basicConfig(level=logging.INFO)

class StreamingOpenCodeInterpreter(OpenCodeInterpreter):
    streamer: Optional[TextIteratorStreamer] = None

    # overwirte generate function
    @torch.inference_mode()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens = 1024,
        do_sample: bool = False,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        # Get the model and tokenizer, and tokenize the user text.

        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, Timeout=5
        )

        inputs = self.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKEN_LENGTH)
        inputs = inputs.to(self.model.device)

        kwargs = dict(
            **inputs,
            streamer=self.streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        return ""

def save_json(dialog, mode, json_file_path, dialog_id) -> None:
    with scheduler.lock:
        with json_file_path.open("a") as f:
            json.dump({"id": dialog_id, "dialog": dialog, "mode": mode, "datetime": datetime.now().isoformat()}, f, ensure_ascii=False)
            f.write("\n")

def convert_history(gradio_history: list[list], interpreter_history: list[dict]):
    interpreter_history = [interpreter_history[0]] if interpreter_history and interpreter_history[0]["role"] == "system" else []
    if not gradio_history:
        return interpreter_history
    for item in gradio_history:
        if item[0] is not None:
            interpreter_history.append({"role": "user", "content": item[0]})
        if item[1] is not None:
            interpreter_history.append({"role": "assistant", "content": item[1]})
    return interpreter_history

def update_uuid(dialog_info):
    new_uuid = str(uuid4())
    logging.info(f"allocating new uuid {new_uuid} for conversation...")
    return [new_uuid, dialog_info[1]]

def is_valid_python_code(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class InputFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found_input = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'input':
            self.found_input = True
        self.generic_visit(node)

def has_input_function_calls(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    visitor = InputFunctionVisitor()
    visitor.visit(tree)
    return visitor.found_input

def gradio_launch(model_path: str, MAX_TRY: int = 3):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(height=600, label="OpenCodeInterpreter", avatar_images=["assets/user.pic.jpg", "assets/assistant.pic.jpg"], show_copy_button=True)
        with gr.Group():
            with gr.Row():
                msg = gr.Textbox(
                    container=False,
                    show_label=False,
                    label="Message",
                    placeholder="Type a message...",
                    scale=7,
                    autofocus=True
                )
                sub = gr.Button(
                    "Submit",
                    variant="primary",
                    scale=1,
                    min_width=150
                )
                # stop = gr.Button(
                #     "Stop",
                #     variant="stop",
                #     visible=False,
                #     scale=1,
                #     min_width=150
                # )

        with gr.Row():
            # retry = gr.Button("🔄  Retry", variant="secondary")
            # undo = gr.Button("↩️ Undo", variant="secondary")
            clear = gr.Button("🗑️  Clear", variant="secondary")

        session_state = gr.State([])
        jupyter_state = gr.State(JupyterNotebook())
        dialog_info = gr.State(["", 0])
        demo.load(update_uuid, dialog_info, dialog_info)

        def bot(user_message, history, jupyter_state, dialog_info, interpreter):
            logging.info(f"user message: {user_message}")
            interpreter.dialog = convert_history(gradio_history=history, interpreter_history=interpreter.dialog)
            history.append([user_message, None])

            interpreter.dialog.append({"role": "user", "content": user_message})

            # setup
            HAS_CODE = False  # For now
            prompt = interpreter.dialog_to_prompt(dialog=interpreter.dialog)

            _ = interpreter.generate(prompt)
            history[-1][1] = ""
            generated_text = ""
            for character in interpreter.streamer:
                history[-1][1] += character
                history[-1][1] = history[-1][1].replace("<|EOT|>","")
                generated_text += character
                yield history, history, jupyter_state, dialog_info

            if is_valid_python_code(history[-1][1].strip()):
                history[-1][1] = f"```python\n{history[-1][1].strip()}\n```"
                generated_text = history[-1][1]

            HAS_CODE, generated_code_block = interpreter.extract_code_blocks(
                generated_text
            )

            interpreter.dialog.append(
                {
                    "role": "assistant",
                    "content": generated_text.replace("<unk>_", "")
                    .replace("<unk>", "")
                    .replace("<|EOT|>", ""),
                }
            )

            logging.info(f"saving current dialog to file {dialog_info[0]}.json...")
            logging.info(f"current dialog: {interpreter.dialog}")
            save_json(interpreter.dialog, mode="openci_only", json_file_path=JSON_DATASET_DIR/f"{dialog_info[0]}.json", dialog_id=dialog_info[0])

            attempt = 1
            while HAS_CODE:
                if attempt > MAX_TRY:
                    break
                # if no code then doesn't have to execute it
                generated_text = "" # clear generated text

                yield history, history, jupyter_state, dialog_info

                # replace unknown thing to none ''
                generated_code_block = generated_code_block.replace(
                    "<unk>_", ""
                ).replace("<unk>", "")

                if has_input_function_calls(generated_code_block):
                    code_block_output = "Please directly assign the value of inputs instead of using input() function in your code."
                else:
                    (
                        code_block_output,
                        error_flag,
                    ) = interpreter.execute_code_and_return_output(
                        f"{generated_code_block}",
                        jupyter_state
                    )
                    if error_flag == "Timeout":
                        logging.info(f"{dialog_info[0]}: Restart jupyter kernel due to timeout")
                        jupyter_state = JupyterNotebook()
                    code_block_output = interpreter.clean_code_output(code_block_output)

                    if code_block_output.strip():
                        code_block_output = "Execution result: \n" + code_block_output
                    else:
                        code_block_output = "Code is executed, but result is empty. Please make sure that you include test case in your code."

                history.append([code_block_output, ""])

                interpreter.dialog.append({"role": "user", "content": code_block_output})

                yield history, history, jupyter_state, dialog_info

                prompt = interpreter.dialog_to_prompt(dialog=interpreter.dialog)

                logging.info(f"generating answer for dialog {dialog_info[0]}")
                _ = interpreter.generate(prompt)
                for character in interpreter.streamer:
                    history[-1][1] += character
                    history[-1][1] = history[-1][1].replace("<|EOT|>","")
                    generated_text += character
                    yield history, history, jupyter_state, dialog_info
                logging.info(f"finish generating answer for dialog {dialog_info[0]}")

                HAS_CODE, generated_code_block = interpreter.extract_code_blocks(
                    history[-1][1]
                )

                interpreter.dialog.append(
                    {
                        "role": "assistant", 
                        "content": generated_text.replace("<unk>_", "")
                        .replace("<unk>", "")
                        .replace("<|EOT|>", ""),
                    }
                )

                attempt += 1

                logging.info(f"saving current dialog to file {dialog_info[0]}.json...")
                logging.info(f"current dialog: {interpreter.dialog}")
                save_json(interpreter.dialog, mode="openci_only", json_file_path=JSON_DATASET_DIR/f"{dialog_info[0]}.json", dialog_id=dialog_info[0])

                if generated_text.endswith("<|EOT|>"):
                    continue

            return history, history, jupyter_state, dialog_info


        def reset_textbox():
            return gr.update(value="")


        def clear_history(history, jupyter_state, dialog_info, interpreter):
            interpreter.dialog = []
            jupyter_state.close()
            return [], [], JupyterNotebook(), update_uuid(dialog_info)

        interpreter = StreamingOpenCodeInterpreter(model_path=model_path)

        sub.click(partial(bot, interpreter=interpreter), [msg, session_state, jupyter_state, dialog_info], [chatbot, session_state, jupyter_state, dialog_info])
        sub.click(reset_textbox, [], [msg])

        clear.click(partial(clear_history, interpreter=interpreter), [session_state, jupyter_state, dialog_info], [chatbot, session_state, jupyter_state, dialog_info], queue=False)

    demo.queue(max_size=20)
    demo.launch(share=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        help="Path to the OpenCodeInterpreter Model.",
        default="m-a-p/OpenCodeInterpreter-DS-6.7B",
    )
    args = parser.parse_args()

    gradio_launch(model_path=args.path)
