import re
from typing import Tuple, Dict, Optional, Any

def parse_assistant_output(text):
    """
    Given a string like
      "...<think> some reasoning </think><answer> x=5 y=10 </answer>..."
    returns
      ("some reasoning", {"x": 5, "y": 10})
    If tags are missing, returns None for the reasoning and/or an empty dict.
    """
    # 1) Extract the think block
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else None

    # 2) Extract the answer block
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else ""

    # 3) Parse variable=value pairs
    #    Matches things like "x=5", "y = 10", "z=3.14"
    pairs = re.findall(r"(\w+)\s*=\s*([-+]?\d*\.?\d+)", answer_text)
    parsed: Dict[str, Any] = {}
    for var, val_str in pairs:
        # cast to int if no decimal point, else float
        if "." in val_str:
            parsed[var] = float(val_str)
        else:
            parsed[var] = int(val_str)

    return reasoning, parsed

def score_answer(a, ta):
    try:
        s = int(a['x'] == ta['x']) + int(a['y'] == ta['y'])
        return int(s==2)
    except:
        return 0