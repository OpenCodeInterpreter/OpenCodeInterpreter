TOOLS_CODE = """
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os,sys
import re
from datetime import datetime
from sympy import symbols, Eq, solve
import torch 
import requests
from bs4 import BeautifulSoup
import json
import math
import yfinance
import time
"""

write_denial_function = 'lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("Writing to disk operation is not permitted due to safety reasons. Please do not try again!"))'
read_denial_function = 'lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("Reading from disk operation is not permitted due to safety reasons. Please do not try again!"))'
class_denial = """Class Denial:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            return "Using this class is not permitted due to safety reasons. Please do not try again!"
        return method
"""

GUARD_CODE = f"""
import builtins

_original_open = open

def custom_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if 'w' in mode or 'a' in mode or 'x' in mode or '+' in mode:
        raise PermissionError("Writing operation is not permitted due to safety reasons. Please do not try again!")
    return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

builtins.open = custom_open

builtins.exit = {write_denial_function}
builtins.quit = {write_denial_function}

import sys

blocked_modules = ['pathlib', 'glob', 'ctypes']

for module in blocked_modules:
    sys.modules[module] = PermissionError

import os

os.listdir = {read_denial_function}
os.scandir = {read_denial_function}
os.walk = {read_denial_function}
os.stat = {read_denial_function}
os.kill = {write_denial_function}
os.system = {write_denial_function}
os.putenv = {write_denial_function}
os.remove = {write_denial_function}
os.removedirs = {write_denial_function}
os.rmdir = {write_denial_function}
os.fchdir = {write_denial_function}
os.setuid = {write_denial_function}
os.fork = {write_denial_function}
os.forkpty = {write_denial_function}
os.killpg = {write_denial_function}
os.rename = {write_denial_function}
os.renames = {write_denial_function}
os.truncate = {write_denial_function}
os.replace = {write_denial_function}
os.unlink = {write_denial_function}
os.fchmod = {write_denial_function}
os.fchown = {write_denial_function}
os.chmod = {write_denial_function}
os.chown = {write_denial_function}
os.chroot = {write_denial_function}
os.fchdir = {write_denial_function}
os.lchflags = {write_denial_function}
os.lchmod = {write_denial_function}
os.lchown = {write_denial_function}
os.getcwd = {write_denial_function}
os.chdir = {write_denial_function}
os.popen = {write_denial_function}
os.environ = {{}}
os.getenv = {write_denial_function}
builtins.open = {write_denial_function}

import shutil

shutil.rmtree = {write_denial_function}
shutil.move = {write_denial_function}
shutil.chown = {write_denial_function}

import subprocess

subprocess.Popen = {write_denial_function}  # type: ignore

__builtins__["help"] = {write_denial_function}

import sys

sys.modules["ipdb"] = {write_denial_function}
sys.modules["joblib"] = {write_denial_function}
sys.modules["resource"] = {write_denial_function}
sys.modules["psutil"] = {write_denial_function}
sys.modules["tkinter"] = {write_denial_function}

get_ipython().system = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("Sorry, magic command is disabled due to safety reasons. Please do not try again!"))
"""

CODE_INTERPRETER_SYSTEM_PROMPT = """You are an AI code interpreter.
Your goal is to help users do a variety of jobs by executing Python code.

You should:
1. Comprehend the user's requirements carefully & to the letter.
2. Give a brief description for what you plan to do & call the provided function to run code.
3. Provide results analysis based on the execution output.
4. If error occurred, try to fix it.
5. Response in the same language as the user."""
