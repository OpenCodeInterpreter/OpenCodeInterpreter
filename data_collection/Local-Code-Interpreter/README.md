**Read in other language: [中文](README_CN.md).**

# Local-Code-Interpreter
A local implementation of OpenAI's ChatGPT Code Interpreter (Advanced Data Analysis).

## Introduction

OpenAI's Code Interpreter (currently renamed as Advanced Data Analysis) for ChatGPT is a revolutionary feature that allows the execution of Python code within the AI model. However, it execute code within an online sandbox and has certain limitations. In this project, we present Local Code Interpreter – which enables code execution on your local device, offering enhanced flexibility, security, and convenience.
![notebook_gif_demo](example_img/save_to_notebook_demo.gif)

## Key Advantages

- **Custom Environment**: Execute code in a customized environment of your choice, ensuring you have the right packages and settings.

- **Seamless Experience**: Say goodbye to file size restrictions and internet issues while uploading. With Local Code Interpreter, you're in full control.

- **GPT-3.5 Availability**: While official Code Interpreter is only available for GPT-4 model, the Local Code Interpreter offers the flexibility to switch between both GPT-3.5 and GPT-4 models.

- **Enhanced Data Security**: Keep your data more secure by running code locally, minimizing data transfer over the internet.

- **Jupyter Support**: You can save all the code and conversation history in a Jupyter notebook for future use.

## Note
Executing AI-generated code without human review on your own device is not safe. You are responsible for taking measures to protect the security of your device and data (such as using a virtural machine) before launching this program. All consequences caused by using this program shall be borne by youself.

## Usage

### Installation

1. Clone this repository to your local device
   ```shell
   git clone https://github.com/MrGreyfun/Local-Code-Interpreter.git
   cd Local-Code-Interpreter
   ```

2. Install the necessary dependencies. The program has been tested on Windows 10 and CentOS Linux 7.8, with Python 3.9.16. Required packages include:
   ```text
   Jupyter Notebook    6.5.4
   gradio              3.39.0
   openai              0.27.8
   ansi2html           1.8.0
   tiktoken            0.3.3
   Pillow              9.4.0
   ```
   Other systems or package versions may also work. Please note that you should not update the `openai` package to the latest `1.x` version, as it has been rewritten and is not compatible with older versions.
   You can use the following command to directly install the required packages:
   ```shell
   pip install -r requirements.txt
   ```
   For newcomers to Python, we offer a convenient command that installs additional packages commonly used for data processing and analysis:
   ```shell
   pip install -r requirements_full.txt
   ```
### Configuration

1. Create a `config.json` file in the `src` directory, following the examples provided in the `config_example` directory.

2. Configure your API key in the `config.json` file.

Please Note:
1. **Set the `model_name` Correctly**
    This program relies on the function calling capability of the `0613` or newer versions of models:
    - `gpt-3.5-turbo-0613` (and its 16K version)
    - `gpt-3.5-turbo-1106`
    - `gpt-4-0613` (and its 32K version)
    - `gpt-4-1106-preview` 

    Older versions of the models will not work. Note that `gpt-4-vision-preview` lacks support for function calling, therefore, it should not be set as `GPT-4` model. 

    For Azure OpenAI service users:
    - Set the `model_name` as your deployment name.
    - Confirm that the deployed model corresponds to the `0613` or newer version.

2. **API Version Settings**
    If you're using Azure OpenAI service, set the `API_VERSION` to `2023-12-01-preview` in the `config.json` file. Note that API versions older than `2023-07-01-preview` do not support the necessary function calls for this program and `2023-12-01-preview` is recommended as older versions will be deprecated in the near future.

3. **Vision Model Settings**
    Despite the `gpt-4-vision-preview` currently does not support function calling, we have implemented vision input using a non-end-to-end approach. To enable vision input, set `gpt-4-vision-preview` as `GPT-4V` model and set `available` to `true`.  Conversely, setting `available` to `false` to disables vision input when unnecessary, which will remove vision-related system prompts and reduce your API costs.
    ![vision_demo](example_img/vision_example.jpg)
4. **Model Context Window Settings**
    The `model_context_window` field records the context window for each model, which the program uses to slice conversations when they exceed the model's context window capacity. 
    Azure OpenAI service users should manually insert context window information using the model's deployment name in the following format:
    ```json
    "<YOUR-DEPLOYMENT-NAME>": <contex_window (integer)>
    ```
   
    Additionally, when OpenAI introduce new models, you can manually append the new model's context window information using the same format. (We will keep this file updated, but there might be delays)

5. **Alternate API Key Handling**
    If you prefer not to store your API key in the `config.json` file, you can opt for an alternate approach:
    - Leave the `API_KEY` field in `config.json` as an empty string:
        ```json
        "API_KEY": ""
        ```
    - Set the environment variable `OPENAI_API_KEY` with your API key before running the program:
        - On Windows:
        ```shell
        set OPENAI_API_KEY=<YOUR-API-KEY>
        ```
        - On Linux:
        ```shell
        export OPENAI_API_KEY=<YOUR-API-KEY>
        ```

## Getting Started

1. Navigate to the `src` directory.
   ```shell
   cd src
   ```

2. Run the command:
   ```shell
   python web_ui.py
   ```

3. Access the generated link in your browser to start using the Local Code Interpreter.

4. Use the `-n` or `--notebook` option to save the conversation in a Jupyter notebook.
   By default, the notebook is saved in the working directory, but you can add a path to save it elsewhere.
   ```shell
   python web_ui.py -n <path_to_notebook>
   ```

## Example

Imagine uploading a data file and requesting the model to perform linear regression and visualize the data. See how Local Code Interpreter provides a seamless experience:

1. Upload the data and request linear regression:
   ![Example 1](example_img/1.jpg)

2. Encounter an error in the generated code:
   ![Example 2](example_img/2.jpg)

3. ChatGPT automatically checks the data structure and fixes the bug:
   ![Example 3](example_img/3.jpg)

4. The corrected code runs successfully:
   ![Example 4](example_img/4.jpg)

5. The final result meets your requirements:
   ![Example 5](example_img/5.jpg)
   ![Example 6](example_img/6.jpg)
