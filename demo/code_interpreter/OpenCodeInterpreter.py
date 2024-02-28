import sys
import os

prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)

from code_interpreter.BaseCodeInterpreter import BaseCodeInterpreter
from utils.const import *

from typing import List, Tuple, Dict
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class OpenCodeInterpreter(BaseCodeInterpreter):
    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        # build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = self.model.eval()

        self.dialog = []
        self.MAX_CODE_OUTPUT_LENGTH = 1000
        

    def dialog_to_prompt(self, dialog: List[Dict]) -> str:
        full_str = self.tokenizer.apply_chat_template(dialog, tokenize=False)

        return full_str

    def extract_code_blocks(self, prompt: str) -> Tuple[bool, str]:
        pattern = re.escape("```python") + r"(.*?)" + re.escape("```")
        matches = re.findall(pattern, prompt, re.DOTALL)

        if matches:
            # Return the last matched code block
            return True, matches[-1].strip()
        else:
            return False, ""

    def clean_code_output(self, output: str) -> str:
        if self.MAX_CODE_OUTPUT_LENGTH < len(output):
            return (
                output[: self.MAX_CODE_OUTPUT_LENGTH // 5]
                + "\n...(truncated due to length)...\n"
                + output[-self.MAX_CODE_OUTPUT_LENGTH // 5 :]
            )

        return output
