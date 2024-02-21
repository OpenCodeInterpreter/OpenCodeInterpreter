import nbformat
from nbformat import v4 as nbf
import ansi2html
import os
import argparse

# main code
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--notebook", help="Path to the output notebook", default=None, type=str)
args = parser.parse_args()
if args.notebook:
    notebook_path = os.path.join(os.getcwd(), args.notebook)
    base, ext = os.path.splitext(notebook_path)
    if ext.lower() != '.ipynb':
        notebook_path += '.ipynb'
    if os.path.exists(notebook_path):
        print(f'File at {notebook_path} already exists. Please choose a different file name.')
        exit()

# Global variable for code cells
nb = nbf.new_notebook()


def ansi_to_html(ansi_text):
    converter = ansi2html.Ansi2HTMLConverter()
    html_text = converter.convert(ansi_text)
    return html_text


def write_to_notebook():
    if args.notebook:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)


def add_code_cell_to_notebook(code):
    code_cell = nbf.new_code_cell(source=code)
    nb['cells'].append(code_cell)
    write_to_notebook()


def add_code_cell_output_to_notebook(output):
    html_content = ansi_to_html(output)
    cell_output = nbf.new_output(output_type='display_data', data={'text/html': html_content})
    nb['cells'][-1]['outputs'].append(cell_output)
    write_to_notebook()


def add_code_cell_error_to_notebook(error):
    nbf_error_output = nbf.new_output(
        output_type='error',
        ename='Error',
        evalue='Error message',
        traceback=[error]
    )
    nb['cells'][-1]['outputs'].append(nbf_error_output)
    write_to_notebook()


def add_image_to_notebook(image, mime_type):
    image_output = nbf.new_output(output_type='display_data', data={mime_type: image})
    nb['cells'][-1]['outputs'].append(image_output)
    write_to_notebook()


def add_markdown_to_notebook(content, title=None):
    if title:
        content = "##### " + title + ":\n" + content
    markdown_cell = nbf.new_markdown_cell(content)
    nb['cells'].append(markdown_cell)
    write_to_notebook()
