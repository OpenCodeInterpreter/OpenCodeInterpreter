import json
import copy
import shutil
from jupyter_backend import *
from tools import *
from typing import *
from notebook_serializer import add_markdown_to_notebook, add_code_cell_to_notebook

functions = [
    {
        "name": "execute_code",
        "description": "This function allows you to execute Python code and retrieve the terminal output. If the code "
                       "generates image output, the function will return the text '[image]'. The code is sent to a "
                       "Jupyter kernel for execution. The kernel will remain active after execution, retaining all "
                       "variables in memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code text"
                }
            },
            "required": ["code"],
        }
    },
]

system_msg = '''You are an AI code interpreter.
Your goal is to help users do a variety of jobs by executing Python code.

You should:
1. Comprehend the user's requirements carefully & to the letter.
2. Give a brief description for what you plan to do & call the provided function to run code.
3. Provide results analysis based on the execution output.
4. If error occurred, try to fix it.
5. Response in the same language as the user.

Note: If the user uploads a file, you will receive a system message "User uploaded a file: filename". Use the filename as the path in the code. '''

with open('config.json') as f:
    config = json.load(f)

if not config['API_KEY']:
    config['API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.unsetenv('OPENAI_API_KEY')


def get_config():
    return config


def config_openai_api(api_type, api_base, api_version, api_key):
    openai.api_type = api_type
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key


class GPTResponseLog:
    def __init__(self):
        self.assistant_role_name = ''
        self.content = ''
        self.function_name = None
        self.function_args_str = ''
        self.code_str = ''
        self.display_code_block = ''
        self.finish_reason = 'stop'
        self.bot_history = None
        self.stop_generating = False
        self.code_executing = False
        self.interrupt_signal_sent = False

    def reset_gpt_response_log_values(self, exclude=None):
        if exclude is None:
            exclude = []

        attributes = {'assistant_role_name': '',
                      'content': '',
                      'function_name': None,
                      'function_args_str': '',
                      'code_str': '',
                      'display_code_block': '',
                      'finish_reason': 'stop',
                      'bot_history': None,
                      'stop_generating': False,
                      'code_executing': False,
                      'interrupt_signal_sent': False}

        for attr_name in exclude:
            del attributes[attr_name]
        for attr_name, value in attributes.items():
            setattr(self, attr_name, value)

    def set_assistant_role_name(self, assistant_role_name: str):
        self.assistant_role_name = assistant_role_name

    def add_content(self, content: str):
        self.content += content

    def set_function_name(self, function_name: str):
        self.function_name = function_name

    def copy_current_bot_history(self, bot_history: List):
        self.bot_history = copy.deepcopy(bot_history)

    def add_function_args_str(self, function_args_str: str):
        self.function_args_str += function_args_str

    def update_code_str(self, code_str: str):
        self.code_str = code_str

    def update_display_code_block(self, display_code_block):
        self.display_code_block = display_code_block

    def update_finish_reason(self, finish_reason: str):
        self.finish_reason = finish_reason

    def update_stop_generating_state(self, stop_generating: bool):
        self.stop_generating = stop_generating

    def update_code_executing_state(self, code_executing: bool):
        self.code_executing = code_executing

    def update_interrupt_signal_sent(self, interrupt_signal_sent: bool):
        self.interrupt_signal_sent = interrupt_signal_sent


class BotBackend(GPTResponseLog):
    def __init__(self):
        super().__init__()
        self.unique_id = hash(id(self))
        self.jupyter_work_dir = f'cache/work_dir_{self.unique_id}'
        self.tool_log = f'cache/tool_{self.unique_id}.log'
        self.jupyter_kernel = JupyterKernel(work_dir=self.jupyter_work_dir)
        self.gpt_model_choice = "GPT-3.5"
        self.revocable_files = []
        self.system_msg = system_msg
        self.functions = copy.deepcopy(functions)
        self._init_api_config()
        self._init_tools()
        self._init_conversation()
        self._init_kwargs_for_chat_completion()

    def _init_conversation(self):
        first_system_msg = {'role': 'system', 'content': self.system_msg}
        self.context_window_tokens = 0  # num of tokens actually sent to GPT
        self.sliced = False  # whether the conversion is sliced
        if hasattr(self, 'conversation'):
            self.conversation.clear()
            self.conversation.append(first_system_msg)
        else:
            self.conversation: List[Dict] = [first_system_msg]

    def _init_api_config(self):
        self.config = get_config()
        api_type = self.config['API_TYPE']
        api_base = self.config['API_base']
        api_version = self.config['API_VERSION']
        api_key = config['API_KEY']
        config_openai_api(api_type, api_base, api_version, api_key)

    def _init_tools(self):
        self.additional_tools = {}

        tool_datas = get_available_tools(self.config)
        if tool_datas:
            self.system_msg += '\n\nAdditional tools:'

        for tool_data in tool_datas:
            system_prompt = tool_data['system_prompt']
            tool_name = tool_data['tool_name']
            tool_description = tool_data['tool_description']

            self.system_msg += f'\n{tool_name}: {system_prompt}'

            self.functions.append(tool_description)
            self.additional_tools[tool_name] = {
                'tool': tool_data['tool'],
                'additional_parameters': copy.deepcopy(tool_data['additional_parameters'])
            }
            for parameter, value in self.additional_tools[tool_name]['additional_parameters'].items():
                if callable(value):
                    self.additional_tools[tool_name]['additional_parameters'][parameter] = value(self)

    def _init_kwargs_for_chat_completion(self):
        self.kwargs_for_chat_completion = {
            'stream': True,
            'messages': self.conversation,
            'functions': self.functions,
            'function_call': 'auto'
        }

        model_name = self.config['model'][self.gpt_model_choice]['model_name']

        if self.config['API_TYPE'] == 'azure':
            self.kwargs_for_chat_completion['engine'] = model_name
        else:
            self.kwargs_for_chat_completion['model'] = model_name

    def _backup_all_files_in_work_dir(self):
        count = 1
        backup_dir = f'cache/backup_{self.unique_id}'
        while os.path.exists(backup_dir):
            count += 1
            backup_dir = f'cache/backup_{self.unique_id}_{count}'
        shutil.copytree(src=self.jupyter_work_dir, dst=backup_dir)

    def _clear_all_files_in_work_dir(self, backup=True):
        if backup:
            self._backup_all_files_in_work_dir()
        for filename in os.listdir(self.jupyter_work_dir):
            path = os.path.join(self.jupyter_work_dir, filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def _save_tool_log(self, tool_response):
        with open(self.tool_log, 'a', encoding='utf-8') as log_file:
            log_file.write(f'Previous conversion: {self.conversation}\n')
            log_file.write(f'Model choice: {self.gpt_model_choice}\n')
            log_file.write(f'Tool name: {self.function_name}\n')
            log_file.write(f'Parameters: {self.function_args_str}\n')
            log_file.write(f'Response: {tool_response}\n')
            log_file.write('----------\n\n')

    def add_gpt_response_content_message(self):
        self.conversation.append(
            {'role': self.assistant_role_name, 'content': self.content}
        )
        add_markdown_to_notebook(self.content, title="Assistant")

    def add_text_message(self, user_text):
        self.conversation.append(
            {'role': 'user', 'content': user_text}
        )
        self.revocable_files.clear()
        self.update_finish_reason(finish_reason='new_input')
        add_markdown_to_notebook(user_text, title="User")

    def add_file_message(self, path, bot_msg):
        filename = os.path.basename(path)
        work_dir = self.jupyter_work_dir

        shutil.copy(path, work_dir)

        gpt_msg = {'role': 'system', 'content': f'User uploaded a file: {filename}'}
        self.conversation.append(gpt_msg)
        self.revocable_files.append(
            {
                'bot_msg': bot_msg,
                'gpt_msg': gpt_msg,
                'path': os.path.join(work_dir, filename)
            }
        )

    def add_function_call_response_message(self, function_response: Union[str, None], save_tokens=True):
        if self.code_str is not None:
            add_code_cell_to_notebook(self.code_str)

        self.conversation.append(
            {
                "role": self.assistant_role_name,
                "name": self.function_name,
                "content": self.function_args_str
            }
        )
        if function_response is not None:
            if save_tokens and len(function_response) > 500:
                function_response = f'{function_response[:200]}\n[Output too much, the middle part output is omitted]\n ' \
                                    f'End part of output:\n{function_response[-200:]}'
            self.conversation.append(
                {
                    "role": "function",
                    "name": self.function_name,
                    "content": function_response,
                }
            )
        self._save_tool_log(tool_response=function_response)

    def append_system_msg(self, prompt):
        self.conversation.append(
            {'role': 'system', 'content': prompt}
        )

    def revoke_file(self):
        if self.revocable_files:
            file = self.revocable_files[-1]
            bot_msg = file['bot_msg']
            gpt_msg = file['gpt_msg']
            path = file['path']

            assert self.conversation[-1] is gpt_msg
            del self.conversation[-1]

            os.remove(path)

            del self.revocable_files[-1]

            return bot_msg
        else:
            return None

    def update_gpt_model_choice(self, model_choice):
        self.gpt_model_choice = model_choice
        self._init_kwargs_for_chat_completion()

    def update_token_count(self, num_tokens):
        self.__setattr__('context_window_tokens', num_tokens)

    def update_sliced_state(self, sliced):
        self.__setattr__('sliced', sliced)

    def send_interrupt_signal(self):
        self.jupyter_kernel.send_interrupt_signal()
        self.update_interrupt_signal_sent(interrupt_signal_sent=True)

    def restart(self):
        self.revocable_files.clear()
        self._init_conversation()
        self.reset_gpt_response_log_values()
        self.jupyter_kernel.restart_jupyter_kernel()
        self._clear_all_files_in_work_dir()
