import jupyter_client
import re


def delete_color_control_char(string):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', string)


class JupyterKernel:
    def __init__(self, work_dir):
        self.kernel_manager, self.kernel_client = jupyter_client.manager.start_new_kernel(kernel_name='python3')
        self.work_dir = work_dir
        self.interrupt_signal = False
        self._create_work_dir()
        self.available_functions = {
            'execute_code': self.execute_code,
            'python': self.execute_code
        }

    def execute_code_(self, code):
        msg_id = self.kernel_client.execute(code)

        # Get the output of the code
        msg_list = []
        while True:
            try:
                iopub_msg = self.kernel_client.get_iopub_msg(timeout=1)
                msg_list.append(iopub_msg)
                if iopub_msg['msg_type'] == 'status' and iopub_msg['content'].get('execution_state') == 'idle':
                    break
            except:
                if self.interrupt_signal:
                    self.kernel_manager.interrupt_kernel()
                    self.interrupt_signal = False
                continue

        all_output = []
        for iopub_msg in msg_list:
            if iopub_msg['msg_type'] == 'stream':
                if iopub_msg['content'].get('name') == 'stdout':
                    output = iopub_msg['content']['text']
                    all_output.append(('stdout', output))
            elif iopub_msg['msg_type'] == 'execute_result':
                if 'data' in iopub_msg['content']:
                    if 'text/plain' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/plain']
                        all_output.append(('execute_result_text', output))
                    if 'text/html' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/html']
                        all_output.append(('execute_result_html', output))
                    if 'image/png' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/png']
                        all_output.append(('execute_result_png', output))
                    if 'image/jpeg' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/jpeg']
                        all_output.append(('execute_result_jpeg', output))
            elif iopub_msg['msg_type'] == 'display_data':
                if 'data' in iopub_msg['content']:
                    if 'text/plain' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/plain']
                        all_output.append(('display_text', output))
                    if 'text/html' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/html']
                        all_output.append(('display_html', output))
                    if 'image/png' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/png']
                        all_output.append(('display_png', output))
                    if 'image/jpeg' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/jpeg']
                        all_output.append(('display_jpeg', output))
            elif iopub_msg['msg_type'] == 'error':
                if 'traceback' in iopub_msg['content']:
                    output = '\n'.join(iopub_msg['content']['traceback'])
                    all_output.append(('error', output))

        return all_output

    def execute_code(self, code):
        text_to_gpt = []
        content_to_display = self.execute_code_(code)
        for mark, out_str in content_to_display:
            if mark in ('stdout', 'execute_result_text', 'display_text'):
                text_to_gpt.append(out_str)
            elif mark in ('execute_result_png', 'execute_result_jpeg', 'display_png', 'display_jpeg'):
                text_to_gpt.append('[image]')
            elif mark == 'error':
                text_to_gpt.append(delete_color_control_char(out_str))

        return '\n'.join(text_to_gpt), content_to_display

    def _create_work_dir(self):
        # set work dir in jupyter environment
        init_code = f"import os\n" \
                    f"if not os.path.exists('{self.work_dir}'):\n" \
                    f"    os.mkdir('{self.work_dir}')\n" \
                    f"os.chdir('{self.work_dir}')\n" \
                    f"del os"
        self.execute_code_(init_code)

    def send_interrupt_signal(self):
        self.interrupt_signal = True

    def restart_jupyter_kernel(self):
        self.kernel_client.shutdown()
        self.kernel_manager, self.kernel_client = jupyter_client.manager.start_new_kernel(kernel_name='python3')
        self.interrupt_signal = False
        self._create_work_dir()
