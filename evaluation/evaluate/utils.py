import os
from evalplus.sanitize import sanitize
from typing import Any, Dict, List, Optional, Tuple, Union
import itertools
import multiprocessing
import time
from multiprocessing import Array, Value
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pickle

from evalplus.data.utils import CACHE_DIR
from evalplus.eval import *
from evalplus.gen.util import trusted_exec
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS, _poly
from evalplus.eval.utils import TimeoutException
from evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)
import re
import resource
import traceback


SUCCESS = "success"
FAILED = "failed"
TIMEOUT = "timed out"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: SUCCESS, _FAILED: FAILED, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

class MyCustomException(BaseException):
    def __init__(self, message):
        self.message = message


def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Computing expected output...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output

def remove_unindented_lines(code, protect_before, execeptions, trim_tails):
    lines = code.splitlines()
    cut_idx = []
    cut_enabled = False
    for i, line in enumerate(lines):
        if not cut_enabled and line.startswith(protect_before):
            cut_enabled = True
            continue
        if line.strip() == "":
            continue
        if any(line.startswith(e) for e in execeptions):
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            cut_idx.append(i)

        if any(line.rstrip().startswith(t) for t in trim_tails):
            # cut off everything behind
            cut_idx.extend(list(range(i, len(lines))))
            break

    return "\n".join([line for i, line in enumerate(lines) if i not in cut_idx])


def to_four_space_indents(old_code):
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code


def sanitize_solution(solution,eofs):
    task_id = solution["task_id"]
    dbg_identifier = task_id
    entry_point = solution["entry_point"]

    old_code = solution["solution"]
    old_code = old_code.strip()

    new_code = sanitize(
        old_code=old_code,
        entry_point=entry_point,
        rm_prefix_lines=None,
        eofs=eofs,
    ).strip()

    # if changed, print the message
    if new_code != old_code:
        msg = "Sanitized: " + dbg_identifier
        print(msg)
    solution["solution"]=new_code
    return solution



################################eval funcs############################
# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]

def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected: List,
    time_limits,
    atol,
    fast_check,
    stat: Value,
    details: Array,
    progress: Value,
    feedback: Value,
    feedback_size: int,
):
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        maximum_memory_bytes = None
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
        exec_globals = {}
        try:
            with swallow_io():
                exec(code, exec_globals)
                try:
                    fn = exec_globals[entry_point]
                except KeyError as e:
                    raise f"Please rename your function to {entry_point} as there is no function named {entry_point}."
                except BaseException as e:
                    raise MyCustomException("An error occurred.")
                for i, inp in enumerate(inputs):
                    # try:
                    with time_limit(time_limits[i]):
                        out = fn(*inp)

                    exp = expected[i]
                    exact_match = out == exp

                    # ================================================ #
                    # ============== special oracles ================= #
                    if dataset == "mbpp":
                        if (
                            "are_equivalent" == entry_point
                        ):  # Mbpp/164 special oracle
                            exact_match = exact_match or True
                        elif "sum_div" == entry_point:  # Mbpp/295 special oracle
                            exact_match = exact_match or out == 0
                        elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                            if isinstance(out, bool):
                                exact_match = out == exp
                            else:
                                exact_match = exp == (out is not None)

                    if dataset == "humaneval":
                        if "find_zero" == entry_point:
                            assert _poly(*out, inp) <= atol, f"The results aren't as expected.\nInput: {inp}\nExpected Output: {exp}\nActual Output: {out}"
                    # ============== special oracles ================= #
                    if atol == 0 and is_floats(exp):
                        atol = 1e-6  # enforce atol for float comparison
                    
                    if not exact_match and atol != 0:
                        try:
                            np.testing.assert_allclose(out, exp, atol=atol)
                        except BaseException as e:
                            raise AssertionError(f"The results aren't as expected.\nInput: {inp}\nExpected Output: {exp}\nActual Output: {out}")
                    else:
                        assert exact_match, f"The results aren't as expected.\nInput: {inp}\nExpected Output: {exp}\nActual Output: {out}"

                    details[i] = True
                    progress.value += 1
            stat.value = _SUCCESS
            padding = feedback_size - len(SUCCESS)
            feedback.value = (SUCCESS + " " * padding).encode('utf-8')
        except TimeoutException as e:
            stat.value = _FAILED
            error_str="Execution timed out."
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8')
        except AssertionError as e:
            stat.value = _FAILED
            error_str=str(e)[:feedback_size]
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8')
        except MyCustomException as e:
            stat.value = _FAILED
            error_str=e.message[:feedback_size]
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8') 
        except BaseException as e:
            stat.value = _FAILED
            error_traceback = traceback.format_exc()
            match = re.search(r'(File "<string>".*)', error_traceback, re.DOTALL)
            if match:
                error_traceback = match.group(1)
            elif "assert _poly" in error_traceback:
                if "TypeError: _poly() argument after *" in error_traceback:
                    error_traceback = "TypeError: Invalid output type, output must be an iterable."
                else:
                    delimiter = r'f"Input: \{inp\}\\nExpected Output: \{exp\}\\nActual Output: \{out\}"'
                    error_traceback = re.split(delimiter, error_traceback)[-1]

            error_str=str(error_traceback)[:feedback_size]
            padding = max(0, feedback_size - len(error_str))
            feedback.value = (error_str + " " * padding).encode('utf-8')
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    dataset: str,
    code: str,
    inputs: List[Any],
    entry_point: str,
    expected,
    atol,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> Tuple[str, np.ndarray]:

    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = sum(time_limits) + 1
    if not fast_check:
        timeout += 1  # extra time for data collection

    # shared memory objects
    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])
    feedback_size = 500
    feedback = Array('c', b'\0' * feedback_size)

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            dataset,
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            fast_check,
            stat,
            details,
            progress,
            feedback,
            feedback_size,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    details = details[: progress.value]
    feedback = feedback.value.decode("utf-8").strip()
    if entry_point not in code:
        feedback = f"Please rename your function to {entry_point} as there is no function named {entry_point}."

    if not stat:
        stat = TIMEOUT

    if stat == SUCCESS:
        if len(details) != len(inputs) or not all(details):
            stat = FAILED

    return stat, details, feedback


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    version="base",
    fast_check=False,
    identifier=None,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> Dict[str, Union[int, Optional[Result]]]:
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
    }
    
    ret["base"] = untrusted_check(
        dataset,
        solution,
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )
    if version=="plus":
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
    return ret