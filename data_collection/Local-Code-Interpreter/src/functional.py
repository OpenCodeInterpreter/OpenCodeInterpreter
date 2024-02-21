from bot_backend import *
import base64
import time
import tiktoken
from notebook_serializer import add_code_cell_error_to_notebook, add_image_to_notebook, add_code_cell_output_to_notebook

SLICED_CONV_MESSAGE = "[Rest of the conversation has been omitted to fit in the context window]"


def get_conversation_slice(conversation, model, encoding_for_which_model, min_output_tokens_count=500):
    """
    Function to get a slice of the conversation that fits in the model's context window. returns: The conversation
    with the first message(explaining the role of the assistant) + the last x messages that can fit in the context
    window.
    """
    encoder = tiktoken.encoding_for_model(encoding_for_which_model)
    count_tokens = lambda txt: len(encoder.encode(txt))
    nb_tokens = count_tokens(conversation[0]['content'])
    sliced_conv = [conversation[0]]
    context_window_limit = int(config['model_context_window'][model])
    max_tokens = context_window_limit - count_tokens(SLICED_CONV_MESSAGE) - min_output_tokens_count
    sliced = False
    for message in conversation[-1:0:-1]:
        nb_tokens += count_tokens(message['content'])
        if nb_tokens > max_tokens:
            sliced_conv.insert(1, {'role': 'system', 'content': SLICED_CONV_MESSAGE})
            sliced = True
            break
        sliced_conv.insert(1, message)
    return sliced_conv, nb_tokens, sliced


def chat_completion(bot_backend: BotBackend):
    model_choice = bot_backend.gpt_model_choice
    model_name = bot_backend.config['model'][model_choice]['model_name']
    kwargs_for_chat_completion = copy.deepcopy(bot_backend.kwargs_for_chat_completion)
    if bot_backend.config['API_TYPE'] == "azure":
        kwargs_for_chat_completion['messages'], nb_tokens, sliced = \
            get_conversation_slice(
                conversation=kwargs_for_chat_completion['messages'],
                model=model_name,
                encoding_for_which_model='gpt-3.5-turbo' if model_choice == 'GPT-3.5' else 'gpt-4'
            )
    else:
        kwargs_for_chat_completion['messages'], nb_tokens, sliced = \
            get_conversation_slice(
                conversation=kwargs_for_chat_completion['messages'],
                model=model_name,
                encoding_for_which_model=model_name
            )

    bot_backend.update_token_count(num_tokens=nb_tokens)
    bot_backend.update_sliced_state(sliced=sliced)

    assert config['model'][model_choice]['available'], f"{model_choice} is not available for your API key"

    assert model_name in config['model_context_window'], \
        f"{model_name} lacks context window information. Please check the config.json file."

    response = openai.ChatCompletion.create(**kwargs_for_chat_completion)
    return response


def add_code_execution_result_to_bot_history(content_to_display, history, unique_id):
    images, text = [], []

    # terminal output
    error_occurred = False

    for mark, out_str in content_to_display:
        if mark in ('stdout', 'execute_result_text', 'display_text'):
            text.append(out_str)
            add_code_cell_output_to_notebook(out_str)
        elif mark in ('execute_result_png', 'execute_result_jpeg', 'display_png', 'display_jpeg'):
            if 'png' in mark:
                images.append(('png', out_str))
                add_image_to_notebook(out_str, 'image/png')
            else:
                add_image_to_notebook(out_str, 'image/jpeg')
                images.append(('jpg', out_str))
        elif mark == 'error':
            # Set output type to error
            text.append(delete_color_control_char(out_str))
            error_occurred = True
            add_code_cell_error_to_notebook(out_str)
    text = '\n'.join(text).strip('\n')
    if error_occurred:
        history.append([None, f'❌Terminal output:\n```shell\n\n{text}\n```'])
    else:
        history.append([None, f'✔️Terminal output:\n```shell\n{text}\n```'])

    # image output
    for filetype, img in images:
        image_bytes = base64.b64decode(img)
        temp_path = f'cache/temp_{unique_id}'
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        path = f'{temp_path}/{hash(time.time())}.{filetype}'
        with open(path, 'wb') as f:
            f.write(image_bytes)
        width, height = get_image_size(path)
        history.append(
            [
                None,
                f'<img src=\"file={path}\" style=\'{"" if width < 800 else "width: 800px;"} max-width:none; '
                f'max-height:none\'> '
            ]
        )


def add_function_response_to_bot_history(hypertext_to_display, history):
    if hypertext_to_display is not None:
        if history[-1][1]:
            history.append([None, hypertext_to_display])
        else:
            history[-1][1] = hypertext_to_display


def parse_json(function_args: str, finished: bool):
    """
    GPT may generate non-standard JSON format string, which contains '\n' in string value, leading to error when using
    `json.loads()`.
    Here we implement a parser to extract code directly from non-standard JSON string.
    :return: code string if successfully parsed otherwise None
    """
    parser_log = {
        'met_begin_{': False,
        'begin_"code"': False,
        'end_"code"': False,
        'met_:': False,
        'met_end_}': False,
        'met_end_code_"': False,
        "code_begin_index": 0,
        "code_end_index": 0
    }
    try:
        for index, char in enumerate(function_args):
            if char == '{':
                parser_log['met_begin_{'] = True
            elif parser_log['met_begin_{'] and char == '"':
                if parser_log['met_:']:
                    if finished:
                        parser_log['code_begin_index'] = index + 1
                        break
                    else:
                        if index + 1 == len(function_args):
                            return None
                        else:
                            temp_code_str = function_args[index + 1:]
                            if '\n' in temp_code_str:
                                try:
                                    return json.loads(function_args + '"}')['code']
                                except json.JSONDecodeError:
                                    try:
                                        return json.loads(function_args + '}')['code']
                                    except json.JSONDecodeError:
                                        try:
                                            return json.loads(function_args)['code']
                                        except json.JSONDecodeError:
                                            if temp_code_str[-1] in ('"', '\n'):
                                                return None
                                            else:
                                                return temp_code_str.strip('\n')
                            else:
                                return json.loads(function_args + '"}')['code']
                elif parser_log['begin_"code"']:
                    parser_log['end_"code"'] = True
                else:
                    parser_log['begin_"code"'] = True
            elif parser_log['end_"code"'] and char == ':':
                parser_log['met_:'] = True
            else:
                continue
        if finished:
            for index, char in enumerate(function_args[::-1]):
                back_index = -1 - index
                if char == '}':
                    parser_log['met_end_}'] = True
                elif parser_log['met_end_}'] and char == '"':
                    parser_log['code_end_index'] = back_index - 1
                    break
                else:
                    continue
            code_str = function_args[parser_log['code_begin_index']: parser_log['code_end_index'] + 1]
            if '\n' in code_str:
                return code_str.strip('\n')
            else:
                return json.loads(function_args)['code']

    except Exception as e:
        return None


def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height
