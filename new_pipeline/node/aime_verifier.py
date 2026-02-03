import re
from typing import Tuple, Optional
from math_verify import parse, verify

_BOXED_RE = re.compile(r"""\\boxed\s*(?:\{([^{}]*)\}|\(([^()]*)\))""", re.DOTALL)

def _clean_latex_scalar(s: str) -> str:
    """
    Make a best-effort to turn a short LaTeX-ish scalar into a plain string.
    For AIME-style answers (0–999), prefer the last integer if present.
    """
    s = s.strip()
    # strip inline math fences like $...$ or \( ... \) or \[ ... \]
    s = re.sub(r"^\$|^\s*\\\(|^\s*\\\[|\$\s*$|\\\)\s*$|\\\]\s*$", "", s).strip()
    # If there's a standalone small integer at the end, use it
    m = re.search(r"(-?\d{1,5})(?!\d)", s)
    if m:
        return m.group(1)
    return s

def _find_last_boxed(text: str) -> Optional[str]:
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    # One of the two groups will be None depending on { } vs ( )
    content = last.group(1) if last.group(1) is not None else last.group(2)
    return _clean_latex_scalar(content or "")

def _fallback_tail(text: str, tail_len: int = 120) -> str:
    tail = text[-tail_len:]
    return _clean_latex_scalar(tail)

def parse_assistant_output(text: str, THINK_TAG) -> Tuple[Optional[str], str]:
    # 1) Extract the think block
    think_match = re.search(r"<" + THINK_TAG + ">(.*?)</" + THINK_TAG + ">", text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else None

    # 2) Extract the answer block
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else ""

    # 3) Fallbacks: last \boxed{...} or tail
    if not answer_text:
        answer_text = _find_last_boxed(text) or _fallback_tail(text)

    # # 4) Extract the $Answer block
    # answer_match = re.search(r"\$Answer (.*)", text, re.DOTALL)
    # answer_text = answer_match.group(1).strip() if answer_match else ""

    return reasoning, answer_text

def score_answer(a, ta):
    try:
        mv_parsed_solution = parse(a)
        mv_gt = parse(ta)
        score = verify(mv_gt, mv_parsed_solution)
        return int(score == 1)
    except:
        return 0