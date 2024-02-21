from functional import *


class ChoiceStrategy(metaclass=ABCMeta):
    def __init__(self, choice):
        self.choice = choice
        self.delta = choice['delta']

    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def execute(self, bot_backend: BotBackend, history: List, whether_exit: bool):
        pass


class RoleChoiceStrategy(ChoiceStrategy):

    def support(self):
        return 'role' in self.delta

    def execute(self, bot_backend: BotBackend, history: List, whether_exit: bool):
        bot_backend.set_assistant_role_name(assistant_role_name=self.delta['role'])
        return history, whether_exit


class ContentChoiceStrategy(ChoiceStrategy):
    def support(self):
        return 'content' in self.delta and self.delta['content'] is not None
        # null value of content often occur in function call:
        #     {
        #       "role": "assistant",
        #       "content": null,
        #       "function_call": {
        #         "name": "python",
        #         "arguments": ""
        #       }
        #     }

    def execute(self, bot_backend: BotBackend, history: List, whether_exit: bool):
        bot_backend.add_content(content=self.delta.get('content', ''))
        history[-1][1] = bot_backend.content
        return history, whether_exit


class NameFunctionCallChoiceStrategy(ChoiceStrategy):
    def support(self):
        return 'function_call' in self.delta and 'name' in self.delta['function_call']

    def execute(self, bot_backend: BotBackend, history: List, whether_exit: bool):
        python_function_dict = bot_backend.jupyter_kernel.available_functions
        additional_tools = bot_backend.additional_tools
        bot_backend.set_function_name(function_name=self.delta['function_call']['name'])
        bot_backend.copy_current_bot_history(bot_history=history)
        if bot_backend.function_name not in python_function_dict and bot_backend.function_name not in additional_tools:
            history.append(
                [
                    None,
                    f'GPT attempted to call a function that does '
                    f'not exist: {bot_backend.function_name}\n '
                ]
            )
            whether_exit = True

        return history, whether_exit


class ArgumentsFunctionCallChoiceStrategy(ChoiceStrategy):

    def support(self):
        return 'function_call' in self.delta and 'arguments' in self.delta['function_call']

    def execute(self, bot_backend: BotBackend, history: List, whether_exit: bool):
        bot_backend.add_function_args_str(function_args_str=self.delta['function_call']['arguments'])

        if bot_backend.function_name == 'python':  # handle hallucinatory function calls
            """
            In practice, we have noticed that GPT, especially GPT-3.5, may occasionally produce hallucinatory
            function calls. These calls involve a non-existent function named `python` with arguments consisting 
            solely of raw code text (not a JSON format).
            """
            temp_code_str = bot_backend.function_args_str
            bot_backend.update_code_str(code_str=temp_code_str)
            bot_backend.update_display_code_block(
                display_code_block="\n🔴Working:\n```python\n{}\n```".format(temp_code_str)
            )
            history = copy.deepcopy(bot_backend.bot_history)
            history[-1][1] += bot_backend.display_code_block
        elif bot_backend.function_name == 'execute_code':
            temp_code_str = parse_json(function_args=bot_backend.function_args_str, finished=False)
            if temp_code_str is not None:
                bot_backend.update_code_str(code_str=temp_code_str)
                bot_backend.update_display_code_block(
                    display_code_block="\n🔴Working:\n```python\n{}\n```".format(
                        temp_code_str
                    )
                )
                history = copy.deepcopy(bot_backend.bot_history)
                history[-1][1] += bot_backend.display_code_block
            else:
                history = copy.deepcopy(bot_backend.bot_history)
                history[-1][1] += bot_backend.display_code_block
        else:
            pass

        return history, whether_exit


class FinishReasonChoiceStrategy(ChoiceStrategy):
    def support(self):
        return self.choice['finish_reason'] is not None

    def execute(self, bot_backend: BotBackend, history: List, whether_exit: bool):

        if bot_backend.content:
            bot_backend.add_gpt_response_content_message()

        bot_backend.update_finish_reason(finish_reason=self.choice['finish_reason'])
        if bot_backend.finish_reason == 'function_call':

            if bot_backend.function_name in bot_backend.jupyter_kernel.available_functions:
                history, whether_exit = self.handle_execute_code_finish_reason(
                    bot_backend=bot_backend, history=history, whether_exit=whether_exit
                )
            else:
                history, whether_exit = self.handle_tool_finish_reason(
                    bot_backend=bot_backend, history=history, whether_exit=whether_exit
                )

        bot_backend.reset_gpt_response_log_values(exclude=['finish_reason'])

        return history, whether_exit

    def handle_execute_code_finish_reason(self, bot_backend: BotBackend, history: List, whether_exit: bool):
        function_dict = bot_backend.jupyter_kernel.available_functions
        try:

            code_str = self.get_code_str(bot_backend)

            bot_backend.update_code_str(code_str=code_str)
            bot_backend.update_display_code_block(
                display_code_block="\n🟢Finished:\n```python\n{}\n```".format(code_str)
            )
            history = copy.deepcopy(bot_backend.bot_history)
            history[-1][1] += bot_backend.display_code_block

            # function response
            bot_backend.update_code_executing_state(code_executing=True)
            text_to_gpt, content_to_display = function_dict[
                bot_backend.function_name
            ](code_str)
            bot_backend.update_code_executing_state(code_executing=False)

            # add function call to conversion
            bot_backend.add_function_call_response_message(function_response=text_to_gpt, save_tokens=True)

            if bot_backend.interrupt_signal_sent:
                bot_backend.append_system_msg(prompt='Code execution is manually stopped by user, no need to fix.')

            add_code_execution_result_to_bot_history(
                content_to_display=content_to_display, history=history, unique_id=bot_backend.unique_id
            )
            return history, whether_exit

        except json.JSONDecodeError:
            history.append(
                [None, f"GPT generate wrong function args: {bot_backend.function_args_str}"]
            )
            whether_exit = True
            return history, whether_exit

        except KeyError as key_error:
            history.append([None, f'Backend key_error: {key_error}'])
            whether_exit = True
            return history, whether_exit

        except Exception as e:
            history.append([None, f'Backend error: {e}'])
            whether_exit = True
            return history, whether_exit

    @staticmethod
    def handle_tool_finish_reason(bot_backend: BotBackend, history: List, whether_exit: bool):
        function_dict = bot_backend.additional_tools
        function_name = bot_backend.function_name
        function = function_dict[function_name]['tool']

        # parser function args
        try:
            kwargs = json.loads(bot_backend.function_args_str)
            kwargs.update(function_dict[function_name]['additional_parameters'])
        except json.JSONDecodeError:
            history.append(
                [None, f"GPT generate wrong function args: {bot_backend.function_args_str}"]
            )
            whether_exit = True
            return history, whether_exit

        else:
            # function response
            function_response, hypertext_to_display = function(**kwargs)

            # add function call to conversion
            bot_backend.add_function_call_response_message(function_response=function_response, save_tokens=False)

            # add hypertext response to bot history
            add_function_response_to_bot_history(hypertext_to_display=hypertext_to_display, history=history)

            return history, whether_exit

    @staticmethod
    def get_code_str(bot_backend):
        if bot_backend.function_name == 'python':
            code_str = bot_backend.function_args_str
        else:
            code_str = parse_json(function_args=bot_backend.function_args_str, finished=True)
            if code_str is None:
                raise json.JSONDecodeError
        return code_str


class ChoiceHandler:
    strategies = [
        RoleChoiceStrategy, ContentChoiceStrategy, NameFunctionCallChoiceStrategy,
        ArgumentsFunctionCallChoiceStrategy, FinishReasonChoiceStrategy
    ]

    def __init__(self, choice):
        self.choice = choice

    def handle(self, bot_backend: BotBackend, history: List, whether_exit: bool):
        for Strategy in self.strategies:
            strategy_instance = Strategy(choice=self.choice)
            if not strategy_instance.support():
                continue
            history, whether_exit = strategy_instance.execute(
                bot_backend=bot_backend,
                history=history,
                whether_exit=whether_exit
            )
        return history, whether_exit


def parse_response(chunk, history: List, bot_backend: BotBackend):
    """
    :return: history, whether_exit
    """
    whether_exit = False
    if chunk['choices']:
        choice = chunk['choices'][0]
        choice_handler = ChoiceHandler(choice=choice)
        history, whether_exit = choice_handler.handle(
            history=history,
            bot_backend=bot_backend,
            whether_exit=whether_exit
        )

    return history, whether_exit
