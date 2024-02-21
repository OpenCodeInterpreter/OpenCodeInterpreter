from response_parser import *
import copy
import json
from tqdm import tqdm
import logging
import argparse
import os

def initialization(state_dict: Dict) -> None:
    if not os.path.exists('cache'):
        os.mkdir('cache')
    if state_dict["bot_backend"] is None:
        state_dict["bot_backend"] = BotBackend()
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

def get_bot_backend(state_dict: Dict) -> BotBackend:
    return state_dict["bot_backend"]

def switch_to_gpt4(state_dict: Dict, whether_switch: bool) -> None:
    bot_backend = get_bot_backend(state_dict)
    if whether_switch:
        bot_backend.update_gpt_model_choice("GPT-4")
    else:
        bot_backend.update_gpt_model_choice("GPT-3.5")

def add_text(state_dict, history, text):
    bot_backend = get_bot_backend(state_dict)
    bot_backend.add_text_message(user_text=text)
    history = history + [[text, None]]
    return history, state_dict

def bot(state_dict, history):
    bot_backend = get_bot_backend(state_dict)
    while bot_backend.finish_reason in ('new_input', 'function_call'):
        if history[-1][1]:
            history.append([None, ""])
        else:
            history[-1][1] = ""
        logging.info("Start chat completion")
        response = chat_completion(bot_backend=bot_backend)
        logging.info(f"End chat completion, response: {response}")

        logging.info("Start parse response")
        history, _ = parse_response(
            chunk=response,
            history=history,
            bot_backend=bot_backend
        )
        logging.info("End parse response")
    return history

def main(state, history, user_input):
    history, state = add_text(state, history, user_input)
    last_history = copy.deepcopy(history)
    first_turn_flag = False
    while True:
        if first_turn_flag:
            switch_to_gpt4(state, False)
            first_turn_flag = False
        else:
            switch_to_gpt4(state, True)
        logging.info("Start bot")
        history = bot(state, history)
        logging.info("End bot")
        print(state["bot_backend"].conversation)
        if last_history == copy.deepcopy(history):
            logging.info("No new response, end conversation")
            conversation = [item for item in state["bot_backend"].conversation if item["content"]]
            return conversation
        else:
            logging.info("New response, continue conversation")
            last_history = copy.deepcopy(history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Initialization")

    state = {"bot_backend": None}
    history = []

    initialization(state)
    switch_to_gpt4(state_dict=state, whether_switch=True)

    logging.info("Start")
    with open(args.input_path, "r") as f:
        instructions = [json.loads(line)["query"] for line in f.readlines()]
    all_history = []
    logging.info(f"{len(instructions)} remaining instructions for {args.input_path}")

    for user_input_index, user_input in enumerate(tqdm(instructions)):
        logging.info(f"Start conversation {user_input_index}")
        conversation = main(state, history, user_input)
        all_history.append(
            {
                "instruction": user_input,
                "conversation": conversation
            }
        )
        with open(f"{args.output_path}", "w") as f:
            json.dump(all_history, f, indent=4, ensure_ascii=False)
        state["bot_backend"].restart()
        