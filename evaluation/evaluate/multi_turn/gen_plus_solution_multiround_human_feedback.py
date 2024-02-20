import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from evalplus.data import (get_human_eval_plus, 
    write_jsonl,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from utils import sanitize_solution,check_correctness,get_groundtruth,SUCCESS
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from copy import deepcopy
from chat_with_gpt import gpt_predict

MAX_TRY = 2

def humaneval_build_instruction(languge: str, question: str):
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

mbpp_build_deepseekcoder_instruction='''You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

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


def build_gpt_prompt(pre_messages: str, problem: str, current_code: str, execution_feedback: str):
    prompt = """You are tasked with providing guidance to a programmer who has drafted a code for a programming problem. 
Your role is to mimic human-like responses and offer suggestions for modifying the code based on the observed execution results.
You should refrain from directly writing code.
Begin by thoroughly examining the existing code and its functionality.
Analyze the @@Execution Result obtained from running the @@Existing Code. Identify any errors, unexpected behavior, or deviations from the expected output.
Consider potential edge cases, optimization opportunities, or alternative approaches based on insights from the @@Execution Result.
Offer guidance in a clear and understandable manner, explaining the rationale behind each suggestion.
Refrain from providing actual code solutions, but instead focus on conceptual modifications or strategies.
Provide constructive feedback to help the programmer improve their coding skills.
Remember, your role is to simulate human-like guidance and expertise in programming without directly implementing solutions.
Please respond in no more than three sentences.

@@Problem
{}

@@Existing Code
{}

@@Execution Result
{}

@@Guidance
""".format(problem.strip(), current_code, execution_feedback)
    return prompt


def generate_multi_round(problem, expected_output, example, lang, tokenizer, model,name,flags):
    pre_messages=[]
    if flags.dataset=="humaneval":
        prompt = humaneval_build_instruction(lang, example['prompt'])
    elif flags.dataset=="mbpp":
        prompt = mbpp_build_deepseekcoder_instruction.strip().format(example['prompt'],"python","python")
    pre_messages.append({"role":"user","content":prompt})

    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        return_tensors="pt"
    ).to(model.device)
    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"
    max_new_tokens=1024
    logging.info(f"max_new_tokens{max_new_tokens}")
    outputs = model.generate(
        inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0,
    )
    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    canonical_solution = example["canonical_solution"]
    solution = {k:v for k,v in example.items()}
    solution["solution"]=output
    sanitized_solution = sanitize_solution(deepcopy(solution),flags.eofs)
    attempt = 1
    judge = False
    modify = False
    code = sanitized_solution["solution"]
    while attempt==1 or sanitized_solution["solution"]!="":
        pre_messages.append({"role":"assistant","content":solution["solution"]})
        args = (
            flags.dataset,
            0,
            problem,
            sanitized_solution["solution"],
            expected_output,
            flags.version,
            True,  # fast_check
            example["task_id"]+f'_{attempt}',
            flags.min_time_limit,
            flags.gt_time_limit_factor,
        )
        result = check_correctness(*args)
        if flags.version=="base" and result["base"][0]==SUCCESS:
            code = sanitized_solution["solution"]
            if attempt==2:
                modify = True
            judge = True
            break
        elif flags.version=="plus" and result["plus"][0]==result["base"][0]==SUCCESS:
            code = sanitized_solution["solution"]
            if attempt==2:
                modify = True
            judge = True
            break
        else:
            attempt += 1    
            if attempt > MAX_TRY:
                code = sanitized_solution["solution"]
                break
            execution_feedback=""
            if flags.version=="base":
                execution_feedback=result["base"][2]
            elif flags.version=="plus":
                if result["base"][0]!=SUCCESS:
                    execution_feedback+=result["base"][2]
                    if "The results aren't as expected." in execution_feedback:
                        if result["plus"][0]!=SUCCESS:
                            execution_feedback+="\n"+result["plus"][2]
                else:
                    execution_feedback=result["plus"][2]
            gpt_messages=[{"role":"system","content":"You are a helpful assistant."}]
            gpt_messages.append({"role":"user","content":build_gpt_prompt(\
                pre_messages, example['prompt'], sanitized_solution["solution"], execution_feedback)})
            gpt_feedback=None
            trytime=0
            while gpt_feedback is None and trytime<5:
                gpt_feedback=gpt_predict(gpt_messages)
                trytime+=1
            if gpt_feedback==None:
                raise BaseException("Please resume.")
            if prompt.endswith("@@ Response"):
                prompt +="""
{}

@@ Instruction
{}
""".format(solution["solution"],gpt_feedback)
            else:
                prompt +="""

@@ Response
{}

@@ Instruction
{}
""".format(solution["solution"],gpt_feedback)        
            pre_messages.append({"role":"user","content":gpt_feedback})
            inputs = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt }],
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                inputs, 
                max_new_tokens=max_new_tokens,#待定，evalplus论文说除了gpt用的1024，其他都用的512
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0,
            )
            output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            solution = {k:v for k,v in example.items()}
            solution["solution"]=output 
            sanitized_solution = sanitize_solution(deepcopy(solution),flags.eofs)

    return code,judge,modify


def gen_solution(args):
    a,b = 0,0
    total_modify = 0
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fail_list=[]
    model_path = args.model
    logging.info(f"model:{model_path}")
    model_name =model_path.replace("/", "_")
    lang = "python"
    os.makedirs(os.path.join(args.output_path,model_name),exist_ok=True)
    output_file = os.path.join(args.output_path,model_name,f"multiround_{args.dataset}_{args.version}_solutions-sanitized.jsonl")
    if not args.resume:
        if os.path.exists(output_file):
            logging.info(f"Old sample jsonl file exists, remove it. {output_file}")
            os.remove(output_file)
    else:
        existing_task_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    if 'task_id' in data:
                        existing_task_ids.add(data['task_id'])
                        a=data["pass_num"]
                        total_modify=data["total_modify"]
                        b=data["total_num"]-a
                        fail_list=data["fail_list"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info("load tokenizer {} from {} over.".format(tokenizer.__class__, model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    modelname=model_path.replace("/", "_")
    if args.dataset=="humaneval":
        problems = get_human_eval_plus()
        examples = problems.items()
        dataset_hash = get_human_eval_plus_hash()
        expected_outputs = get_groundtruth(problems, dataset_hash, [])
    else:
        problems = get_mbpp_plus()
        examples = problems.items()
        dataset_hash = get_mbpp_plus_hash()
        expected_outputs = get_groundtruth(
                problems,
                dataset_hash,
                MBPP_OUTPUT_NOT_NONE_TASKS,
            )
    logging.info("Read {} examples for evaluation over.".format(len(examples)))
    for task_id,example in tqdm(examples, desc='Generating'):
        if args.resume:
            if task_id in existing_task_ids:
                continue
        problem = problems[task_id]
        expected_output = expected_outputs[task_id]
        code,judge,modify = generate_multi_round(problem,expected_output,example, lang, tokenizer, model, modelname,args)
        if modify:
            total_modify += 1
        if judge:
            a += 1
        else:
            b += 1
            fail_list.append(task_id)
        result = a/(a+b)
        single_result = (a-total_modify)/(a+b)
        print ("pass num ",a)
        print ("total num",a+b)
        print ("judge: ",judge)
        print ('modify: '+str(modify))
        print ("num modify: "+str(total_modify))
        print ("fail list: ",fail_list)
        print ('multisound rate: '+str(result))
        print ('singlesound rate: '+str(single_result))
        gen_sample=[dict(task_id=task_id, solution=code, pass_num=a, total_modify=total_modify, total_num=a+b, fail_list=fail_list)]
        write_jsonl(output_file, gen_sample ,append=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model path")
    parser.add_argument('--output_path', type=str, help="output path", default="./multiround_output")
    parser.add_argument('--log_file', type=str, help="log file name", default="gen_humaneval_plus_solution_singleround.log")
    parser.add_argument("--min-time-limit", default=1, type=float)
    parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
    parser.add_argument(
        "--version", required=True, type=str, choices=["base", "plus"]
    )
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp"]
    )
    parser.add_argument('--resume', action="store_true")

    args = parser.parse_args()
    args.eofs=None

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
