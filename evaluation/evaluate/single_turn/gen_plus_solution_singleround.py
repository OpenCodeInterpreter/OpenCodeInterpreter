import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from evalplus.data import get_human_eval_plus,get_mbpp_plus, write_jsonl

def build_humaneval_instruction(languge: str, question: str):
    return '''You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Here is the given code to do completion:
```{}
{}
```
Please continue to complete the function with {} programming language. You are not allowed to modify the given code and do the completion only. 

Please return all completed codes in one code block. 
This code block should be in the following format:
```{}
# Your codes here
```

@@ Response
'''.strip().format(languge.lower(), question.strip(),languge.lower(),languge.lower())


build_mbpp_instruction='''You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Here is the given problem and test examples:
{}

Please use the {} programming language to solve this problem.
Please make sure that your code includes the functions from the test samples and that the input and output formats of these functions match the test samples.

Please return all completed codes in one code block. 
This code block should be in the following format:
```{}
# Your codes here
```

@@ Response
'''


def generate_one(example, lang, tokenizer, model, name, flags):
    if flags.dataset=="humaneval":
        prompt = build_humaneval_instruction(lang, example['prompt'])
    else:
        prompt = example['prompt']
        prompt = build_mbpp_instruction.strip().format(prompt,lang,lang)
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt"
    ).to(model.device)
    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"
    max_new_tokens=1024
    outputs = model.generate(
        inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0,
    )
    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return output

def gen_solution(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_path = args.model
    logging.info(f"model:{model_path}")
    model_name = model_path.replace("/", "_")
    
    lang = "python"
    os.makedirs(os.path.join(args.output_path,model_name),exist_ok=True)
    output_file = os.path.join(args.output_path,model_name,f"single_{args.dataset}_plus_solutions.jsonl")
    if os.path.exists(output_file):
        logging.info(f"Old sample jsonl file exists, remove it. {output_file}")
        os.remove(output_file)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info("load tokenizer {} from {} over.".format(tokenizer.__class__, model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    model_name = model_path.replace("/", "_")
    if args.dataset=="humaneval":
        examples = get_human_eval_plus().items()
    else:
        examples = get_mbpp_plus().items()
    logging.info("Read {} examples for evaluation over.".format(len(examples)))
    for task_id,example in tqdm(examples, desc='Generating'):
        code = generate_one(example, lang, tokenizer, model,model_name,args)
        gen_sample=[dict(task_id=task_id, solution=code)]
        write_jsonl(output_file, gen_sample ,append=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model path")
    parser.add_argument('--output_path', type=str, help="output path", default="./output")
    parser.add_argument('--log_file', type=str, help="log file name")
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp"]
    )
    args = parser.parse_args()
    model_name = args.model.replace("/", "_")
    os.makedirs(os.path.join(args.output_path,model_name),exist_ok=True)
    logfile=os.path.join(args.output_path,model_name,args.log_file)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - \n%(message)s')
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - \n%(message)s')
    file_handler.setFormatter(formatter)  
    logging.getLogger().addHandler(file_handler)

    gen_solution(args)
