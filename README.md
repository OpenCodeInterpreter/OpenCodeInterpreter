# OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement

<p align="center">
<img width="1000px" alt="OpenCodeInterpreter" src="https://opencodeinterpreter.github.io/static/images/figure1.png">
</p>
<p align="center">
  <a href="https://opencodeinterpreter.github.io/">[🏠Homepage]</a> 
  |
  <a href="https://github.com/OpenCodeInterpreter/OpenCodeInterpreter/">[🛠️Code]</a> 
</p>
<hr>

## 🌟 Upcoming Features
- 💡 **Open Sourcing OpenCodeInterpreter-SC2 series Model (based on StarCoder2 base);** 

- 💡 **Open Sourcing OpenCodeInterpreter-GM-7b Model with gemma-7b Base;** 

## 🔔News
🛠️[2024-02-29]: Our official online demo is deployed on HuggingFace Spaces! Take a look at [Demo Page](https://huggingface.co/spaces/m-a-p/OpenCodeInterpreter_demo)

🛠️[2024-02-28]: We have open-sourced the Demo Local Deployment Code with a Setup Guide.

✨[2024-02-26]: We have open-sourced the [OpenCodeInterpreter-DS-1.3b](https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-1.3B) Model.

📘[2024-02-26]: We have open-sourced the [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) Dataset.

🚀[2024-02-23]: We have open-sourced the datasets used in our project named [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback).

🔥[2024-02-19]: We have open-sourced all models in the OpenCodeInterpreter series! We welcome everyone to try out our models and look forward to your participation! 😆



## Introduction
OpenCodeInterpreter is a suite of open-source code generation systems aimed at bridging the gap between large language models and sophisticated proprietary systems like the GPT-4 Code Interpreter. It significantly enhances code generation capabilities by integrating execution and iterative refinement functionalities.

## Models
All models within the OpenCodeInterpreter series have been open-sourced on Hugging Face. You can access our models via the following link: [OpenCodeInterpreter Models](https://huggingface.co/collections/m-a-p/opencodeinterpreter-65d312f6f88da990a64da456).

## Data Collection
Supported by Code-Feedback, a dataset featuring 68K multi-turn interactions, OpenCodeInterpreter incorporates execution and human feedback for dynamic code refinement. 
For additional insights into data collection procedures, please consult the readme provided under [Data Collection](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter/blob/main/data_collection/README.md).

## Evaluation
Our evaluation framework primarily utilizes HumanEval and MBPP, alongside their extended versions, HumanEval+ and MBPP+, leveraging the [EvalPlus framework](https://github.com/evalplus/evalplus) for a more comprehensive assessment.
For specific evaluation methodologies, please refer to the [Evaluation README](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter/blob/main/evaluation/README.md) for more details.

## Demo
We're excited to present our open-source demo, enabling users to effortlessly generate and execute code with our LLM locally. Within the demo, users can leverage the power of LLM to generate code and execute it locally, receiving automated execution feedback. LLM dynamically adjusts the code based on this feedback, ensuring a smoother coding experience. Additionally, users can engage in chat-based interactions with the LLM model, providing feedback to further enhance the generated code.

To begin exploring the demo and experiencing the capabilities firsthand, please refer to the instructions outlined in the [OpenCodeInterpreter Demo README](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter/blob/main/demo/README.md) file. Happy coding!

### Quick Start
- **Entering the workspace**:
   ```bash
   git clone https://github.com/OpenCodeInterpreter/OpenCodeInterpreter.git
   cd demo
   ```
- **Create a new conda environment**: `conda create -n demo python=3.10`

- **Activate the demo environment you create**: `conda activate demo`

- **Install requirements**: `pip install -r requirements.txt`

- **Create a Huggingface access token with write permission [here](https://huggingface.co/docs/hub/en/security-tokens). Our code will only use this token to create and push content to a specific repository called `opencodeinterpreter_user_data` under your own Huggingface account. We cannot get access to your data if you deploy this demo on your own device.**

- **Add the access token to environment variables:** `export HF_TOKEN="your huggingface access token"`

- **Run the Gradio App**:
   ```bash
   python3 chatbot.py --path "the model name of opencodeinterpreter model family. e.g., m-a-p/OpenCodeInterpreter-DS-6.7B"
   ```
### Video
https://github.com/OpenCodeInterpreter/OpenCodeInterpreter/assets/46103100/2337f34d-f5ed-4ecb-857b-3c2d085b72fd


## Contact

If you have any inquiries, please feel free to raise an issue or reach out to us via email at: xiangyue.work@gmail.com, zhengtianyu0428@gmail.com. 
We're here to assist you!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenCodeInterpreter/OpenCodeInterpreter&type=Date)](https://star-history.com/#OpenCodeInterpreter/OpenCodeInterpreter&Date)
