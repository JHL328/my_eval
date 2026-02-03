import logging
import math
import random
import re
import string
from collections.abc import Iterable
from typing import List, Optional, Union

import numpy as np
import sacrebleu

from lm_eval.api.registry import register_aggregation, register_metric


eval_logger = logging.getLogger(__name__)


# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("nanmean")
def nanmean(arr):
    if len(arr) == 0 or all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("harmonic")
def harmonic(arr):
    return len(arr) / (sum([1 / (val + 0.01) for val in arr]))


@register_aggregation("median")
def median(arr):
    return arr[len(arr) // 2]


# Certain metrics must be calculated across all documents in a benchmark.
# We use them as aggregation metrics, paired with no-op passthrough metric fns.
@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_aggregation("f1")
def f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)

    return np.max(fscore)


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


@register_aggregation("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("chrf")
def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_aggregation("brier_score")
def brier_score(items):  # This is a passthrough function
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape

    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score",
)
def brier_score_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


###


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)

@register_metric(
    metric="pass@k",            # use this to call the metric in YAML
    higher_is_better=True,
    # likelihood_required=False,  # only process generated content
    aggregation="mean",         # harness automatically averages over samples
    output_type="generate_until"  # for generate / MC tasks
) 
def pass_at_k(
    *pos_args,                           # match the fallback position arguments
    references: Optional[List[str]] = None,
    predictions: Optional[List[Union[str, List[str]]]] = None,
    args: Optional[dict] = None,        # YAML will pass {...}
    k: int = 16,                        # if args is empty, use default 16
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    regexes_to_ignore: Optional[List[str]] = None,
    **_,
) -> float:
    """
    Parameters
    ----------
    references
        List[str]; len==1 reference answer.
    predictions
        - for `repeats` →   [pred_1, pred_2, ...]  (length >= k)
        - for `repeats` → [[pred_1, pred_2, ...]]
        both formats should be compatible.
    k
        k in pass@k; in YAML `args:{k: ...}` overwrite.
    other kwargs
        same as exact_match.
    """
    gold = references[0].strip()

    # compatible with two predictions structures
    if len(predictions) == 1 and isinstance(predictions[0], list):
        preds = predictions[0]
    else:
        preds = predictions

    # helper same as official exact_match
    def _is_correct(p: str) -> bool:
        return bool(
            exact_match_hf_evaluate(
                predictions=[p.strip()],
                references=[gold],
                ignore_case=ignore_case,
                ignore_punctuation=ignore_punctuation,
                regexes_to_ignore=regexes_to_ignore or [],
            )["exact_match"]
        )

    return {"pass@k": float(any(_is_correct(p) for p in preds[:k]))}


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="bits_per_byte",
)
def bits_per_byte_fn(items):  # This is a passthrough function
    return items


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice", "generate_until"],
    aggregation="bypass",
)
def bypass(items):
    return None


@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="f1",
)
def f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="ter",
)
def ter_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# stderr stuff


class _bootstrap_internal:
    def __init__(self, f, n) -> None:
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print("bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters: int):
    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
        nanmean,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
    assert metrics is not None, (
        "Need to pass a list of each subtask's metric for this stderr aggregation"
    )
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
        curr_score = ((curr_score * curr_size) + (score * size)) / (
            curr_size + size
        )  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (
            curr_size + size - 1
        ) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (
            curr_score - score
        ) ** 2

    return np.sqrt(variance)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)


def aggregate_harmonic_subtask_metrics(metrics, sizes, weight_by_size=True):
    if not weight_by_size:
        sizes = [1] * len(sizes)
    assert len(metrics) == len(sizes)
    return sum(sizes) / (sum([size / (val + 0.01) for size, val in zip(sizes, metrics)]))


@register_metric(
    metric="bigbench_extrahard",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def bbeh_accuracy(predictions, references):

    """Evaluation functions for BigBench Extra Hard."""
    def strip_latex(response: str) -> str:
        if response.startswith("$") and response.endswith("$"):
            response = response[1:-1]
        if "boxed{" in response and response.endswith("}"):
            response = response[0:-1].split("boxed{")[1]
        if "text{" in response and response.endswith("}"):
            response = response[0:-1].split("text{")[1]
        if "texttt{" in response and response.endswith("}"):
            response = response[0:-1].split("texttt{")[1]
        return response

    def extract_answer(sample: str) -> str:
        """Extracts the final answer from the sample."""
        answer_prefixes = [
            "</think>",
            "The answer is:",
            "The final answer is: ",
            "The correct answer is:",
            "The correct answer is",
            "is:\n",
            "The answer is ",
            "**Answer:**",
            "**Final Answer:**",
            "**option ",
            "Answer:",
            "boxed{"
        ]
        answer = sample
        for answer_prefix in answer_prefixes:
            if answer_prefix.lower() in answer.lower():
                answer = answer.split(answer_prefix)[-1].strip()
            if answer.endswith("."):
                answer = answer[:-1]
        return strip_latex(answer)

    def fuzzy_match(prediction: str, reference: str) -> bool:
        import copy
        if not prediction: return False
        prediction = prediction.strip()
        reference = reference.strip()
        import re
        if "**" in prediction:
            for m in re.finditer(r"\*\*(.*)\*\*", prediction):
                if len(m.groups()) > 1 and  len(m[1]) < len(prediction) and fuzzy_match(m[1], reference):
                    return True
        """Fuzzy match function for BigBench Extra Hard."""
        if prediction == reference:
            return True
        # (a) vs a
        if len(prediction) == 3 and prediction[0] == "(" and prediction[2] == ")":
            return prediction[1] == reference
        if len(reference) == 3 and reference[0] == "(" and reference[2] == ")":
            if len(prediction) == 1 or len(prediction) == 2 and prediction[1] == ")":
                return reference[1] == prediction[0]
            elif prediction[0] == reference[1] and (len(prediction) == 1 or not prediction[1].isalnum()):
                return True

        # Numbers
        try:
            if float(prediction) == float(reference):
                return True
        except ValueError:
            pass

        # quote issues
        if prediction.replace("'", "") == reference.replace("'", ""):
            return True

        # Bracket issues
        if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
            return True
        # Question mark issues
        if m := re.match(r"\\text{((\w|\s)*)}", prediction):
            if len(m.groups()) > 1:
                inner = m[1]
                if inner.lower() == reference.lower():
                    return True
        if prediction.endswith("?") and prediction[:-1] == reference:
            return True
        if prediction.lower().startswith(reference.lower()) or reference.lower().startswith(prediction.lower()):
            return True
        if re.sub(r"[^a-z0-9]", "", prediction.lower()) == re.sub(r"[^a-z0-9]", "", reference.lower()):
            return True
        return False 

    def preprocess_sample(sample: str) -> str:
        prediction = extract_answer(sample.strip()).lower()
        prediction = prediction.replace(", ", ",")
        return prediction

    def preprocess_reference(reference: str) -> str:
        reference = reference.strip().lower()
        reference = reference.replace(", ", ",")
        return reference

    def evaluate_correctness(sample: str, reference: str) -> bool:
        prediction = preprocess_sample(sample)
        reference = preprocess_reference(reference)
        return fuzzy_match(prediction, reference)

    correct = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        total += 1
        if evaluate_correctness(prediction, reference):
            correct += 1
    return correct / total
