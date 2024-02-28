import re
import os

PYTHON_PREFIX = os.environ.get("CONDA_PREFIX", "/usr/local")

SITE_PKG_ERROR_PREFIX = f'File {PYTHON_PREFIX}/lib/python3.10/'

def get_error_header(traceback_str):
    lines = traceback_str.split('\n')
    for line in lines:
        if 'Error:' in line:
            return line
    return ''  # Return None if no error message is found

def clean_error_msg(error_str:str =''):
    filtered_error_msg = error_str.__str__().split('An error occurred while executing the following cell')[-1].split("\n------------------\n")[-1]
    raw_error_msg = "".join(filtered_error_msg)

    # Remove escape sequences for colored text
    ansi_escape = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
    error_msg = ansi_escape.sub('', raw_error_msg)
    
    error_str_out = ''
    error_msg_only_cell = error_msg.split(SITE_PKG_ERROR_PREFIX)

    error_str_out += f'{error_msg_only_cell[0]}\n'
    error_header = get_error_header(error_msg_only_cell[-1])
    if error_header not in error_str_out:
        error_str_out += get_error_header(error_msg_only_cell[-1])

    return error_str_out