from collections import Counter
import ast
import re
from check_answer import *

import re, ast
from collections import Counter

_NUMERIC_TOKEN_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')


def _coerce_intlike(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _parse_numeric_token(token: str):
    try:
        if "." in token:
            return float(token)
        return int(token)
    except Exception:
        return None


def _extract_last_intlike_candidate(text: str):
    answer_lines = re.findall(r'(?im)^Answer:\s*(.+)$', text)
    for raw in reversed(answer_lines):
        for token in reversed(_NUMERIC_TOKEN_RE.findall(raw)):
            candidate = _coerce_intlike(_parse_numeric_token(token))
            if candidate is not None:
                return candidate

    for token in reversed(_NUMERIC_TOKEN_RE.findall(text)):
        candidate = _coerce_intlike(_parse_numeric_token(token))
        if candidate is not None:
            return candidate

    return None


def _extract_last_numeric_candidate(text: str):
    answer_lines = re.findall(r'(?im)^Answer:\s*(.+)$', text)
    for raw in reversed(answer_lines):
        tokens = _NUMERIC_TOKEN_RE.findall(raw)
        if tokens:
            return _parse_numeric_token(tokens[-1])

    tokens = _NUMERIC_TOKEN_RE.findall(text)
    if tokens:
        return _parse_numeric_token(tokens[-1])

    return None


def _normalize_int_list(candidate):
    if not isinstance(candidate, (list, tuple)):
        return None

    normalized = []
    for value in candidate:
        coerced = _coerce_intlike(value)
        if coerced is None:
            return None
        normalized.append(coerced)
    return normalized

def extract_coin_change_from_text(text, params):
    """
    Extract & evaluate an integer answer from model output.

    Extraction:
      - Prefer: 'Answer: <int>'
      - Fallback: last integer in the text

    Returns:
      (candidate, True)  if an int is found and equals params["answer"]
      (candidate, False) if an int is found but incorrect
      (None, False)      if no int is found
    
    """
    gt = params.get("answer", None)
    if gt is None:
        raise KeyError('params must contain "answer"')

    m = re.findall(r'Answer:\s*(-?\d+)', text)
    if m:
        cand = int(m[-1])
        return cand, (cand == gt)

    m2 = re.findall(r'-?\d+', text)
    if m2:
        cand = int(m2[-1])
        return cand, (cand == gt)

    return None, False


def extract_knapsack_from_text(text, params):
    """
    Extract & evaluate an integer answer from model output.

    Extraction:
      - Prefer: 'Answer: <int>'
      - Fallback: last integer in the text

    Returns:
      (candidate, True)  if an int is found and equals params["answer"]
      (candidate, False) if an int is found but incorrect
      (None, False)      if no int is found
    
    Note: correctness checks only the max value, not the chosen subset.

    """
    gt = params.get("answer", None)
    if gt is None:
        raise KeyError('params must contain "answer"')

    m = re.findall(r'Answer:\s*(-?\d+)', text)
    if m:
        cand = int(m[-1])
        return cand, (cand == gt)

    m2 = re.findall(r'-?\d+', text)
    if m2:
        cand = int(m2[-1])
        return cand, (cand == gt)

    return None, False


def extract_max_subarray_from_text(text, params):
    """
    Extract & evaluate an integer answer from model output.

    Extraction:
      - Prefer: 'Answer: <int>'
      - Fallback: last integer in the text

    Returns:
      (candidate, True)  if an int is found and equals params["answer"]
      (candidate, False) if an int is found but incorrect
      (None, False)      if no int is found
    
    """
    gt = params.get("answer", None)
    if gt is None:
        raise KeyError('params must contain "answer"')

    m = re.findall(r'Answer:\s*(-?\d+)', text)
    if m:
        cand = int(m[-1])
        return cand, (cand == gt)

    m2 = re.findall(r'-?\d+', text)
    if m2:
        cand = int(m2[-1])
        return cand, (cand == gt)

    return None, False

def _is_subsequence(sub: str, s: str) -> bool:
    it = iter(s)
    for ch in sub:
        for t in it:
            if t == ch:
                break
        else:
            return False
    return True

def extract_lcs_from_text(text, params):
    """
    Extract & evaluate an LCS string from model output.

    Expected params:
      - params["s1"]: str
      - params["s2"]: str
      - params["best_len"]: int

    Correct if:
      - candidate is a string
      - candidate is a subsequence of both s1 and s2
      - len(candidate) == best_len
    """
    s1 = params["s1"]
    s2 = params["s2"]
    best_len = params["best_len"]

    # Preferred: Answer: ...
    m = re.findall(r'Answer:\s*(.*)', text)
    if m:
        raw = m[-1].strip()
        if raw == "":
            cand = ""
        else:
            cand = None
            try:
                v = ast.literal_eval(raw)
                if isinstance(v, str):
                    cand = v
            except Exception:
                pass
            if cand is None:
                cand = raw.strip()
                if len(cand) >= 2 and ((cand[0] == cand[-1] == '"') or (cand[0] == cand[-1] == "'")):
                    cand = cand[1:-1]

        ok = (
            isinstance(cand, str)
            and _is_subsequence(cand, s1)
            and _is_subsequence(cand, s2)
            and len(cand) == best_len
        )
        return cand, ok

    # Fallback: any quoted string
    mq = re.findall(r'(["\'])(.*?)\1', text, flags=re.DOTALL)
    for _, inner in reversed(mq):
        cand = inner
        ok = _is_subsequence(cand, s1) and _is_subsequence(cand, s2) and len(cand) == best_len
        if ok:
            return cand, True
        return cand, False

    return None, False


def extract_four_sum_closest_from_text(text, params):
    """
    Extract & evaluate 4-sum-closest answer from model output (integer target, no exact match).

    Expected params:
      - params["nums"]: list[int]
      - params["target"]: int
      - params["best_diff"]: int  (minimum achievable |sum - target|, guaranteed > 0)

    Correct if:
      - candidate is a list of exactly 4 integers
      - candidate multiset is contained in nums
      - abs(sum(candidate) - target) == best_diff
    """
    nums = params["nums"]
    target = params["target"]
    best_diff = params["best_diff"]

    orig = Counter(nums)

    def is_valid(candidate):
        if not (isinstance(candidate, list) and len(candidate) == 4 and all(isinstance(x, int) for x in candidate)):
            return False
        cand = Counter(candidate)
        if any(cand[v] > orig[v] for v in cand):
            return False
        return abs(sum(candidate) - target) == best_diff

    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)
    if matches:
        s = matches[-1]
        try:
            cand = ast.literal_eval(s)
            if is_valid(cand):
                return cand, True
            if isinstance(cand, list):
                return cand, False
        except Exception:
            pass

    fallback = re.findall(r'\[[^\]]*\]', text)
    for s in reversed(fallback):
        try:
            cand = ast.literal_eval(s)
            if is_valid(cand):
                return cand, True
            if isinstance(cand, list):
                return cand, False
        except Exception:
            continue

    return None, False


def extract_matrix_chain_multiplication_from_text(text, params):
    """
    Extract & evaluate matrix-chain-multiplication minimum cost from model output.

    Expected params:
      - params["answer"]: int (ground-truth minimum scalar multiplications)

    Extraction:
      - Prefer: 'Answer: <int>'
      - Fallback: last integer in the text

    Returns:
      (candidate, True)  if an int is found and equals params["answer"]
      (candidate, False) if an int is found but incorrect
      (None, False)      if no int is found
    """
    gt = params.get("answer", None)
    if gt is None:
        raise KeyError('params must contain "answer" (ground-truth cost)')

    m = re.findall(r'Answer:\s*(-?\d+)', text)
    if m:
        cand = int(m[-1])
        return cand, (cand == gt)

    m2 = re.findall(r'-?\d+', text)
    if m2:
        cand = int(m2[-1])
        return cand, (cand == gt)

    return None, False

def extract_two_sum_closest_from_text(text, params):
    """
    Extract & evaluate two-sum-closest answer from model output.

    Expected params:
      - params["nums"]: list[int]
      - params["target"]: int
      - params["best_diff"]: int

    Returns:
      (candidate, True)  if candidate is a valid pair achieving best_diff
      (candidate, False) if a pair is found but not optimal / invalid
      (None, False)      if no pair is found
    """
    nums = params["nums"]
    target = params["target"]
    best_diff = params["best_diff"]

    orig = Counter(nums)

    def is_valid(pair):
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, int) for x in pair)):
            return False
        cand = Counter(pair)
        if any(cand[v] > orig[v] for v in cand):
            return False
        return abs(sum(pair) - target) == best_diff

    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)
    if matches:
        s = matches[-1]
        try:
            pair = ast.literal_eval(s)
            if is_valid(pair):
                return pair, True
            if isinstance(pair, list):
                return pair, False
        except Exception:
            pass

    fallback = re.findall(r'\[[^\]]*\]', text)
    for s in reversed(fallback):
        try:
            pair = ast.literal_eval(s)
            if is_valid(pair):
                return pair, True
            if isinstance(pair, list):
                return pair, False
        except Exception:
            continue

    return None, False

def extract_k_sum_in_range_from_text(text, params, k: int):
    """
    General correctness checker for k-sum-in-range.

    Expected params:
      - params["nums"]: list[int]
      - params["L"]: int
      - params["R"]: int

    Validation ensures:
      - candidate is a list of integers of length k
      - candidate is a multiset-subset of nums
      - L <= sum(candidate) <= R

    Returns:
      (candidate, True)  if a valid candidate is found
      (candidate, False) if a candidate is found but invalid
      (None, False)      if no candidate list is found
    """
    nums = params["nums"]
    L = params["L"]
    R = params["R"]

    def is_valid(candidate):
        if not (isinstance(candidate, list) and all(isinstance(x, int) for x in candidate)):
            return False
        if len(candidate) != k:
            return False

        orig = Counter(nums)
        cand = Counter(candidate)
        if any(cand[v] > orig[v] for v in cand):
            return False

        s = sum(candidate)
        return L <= s <= R

    # 1) Preferred: "Answer: [...]"
    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)
    if matches:
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if is_valid(candidate):
                return candidate, True
            if isinstance(candidate, list):
                return candidate, False
        except Exception:
            pass

    # 2) Fallback: any bracketed list
    fallback_matches = re.findall(r'\[[^\]]*\]', text)
    for candidate_str in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(candidate_str)
            if is_valid(candidate):
                return candidate, True
            if isinstance(candidate, list):
                return candidate, False
        except Exception:
            continue

    return None, False

def extract_three_sum_in_range_from_text(text, params):
    return extract_k_sum_in_range_from_text(text, params, k=3)

def extract_four_sum_in_range_from_text(text, params):
    return extract_k_sum_in_range_from_text(text, params, k=4)

def extract_k_sum_from_text(text, params, k: int | None = None):
    """
    Extract & evaluate k-sum (e.g., 3-sum / 4-sum) answer from model output.

    Expected params:
      - params["nums"]: list[int]
      - params["target"] or params["target_sum"]: int
      - params["k"] (optional): int, if k is not provided in args

    Return:
      (candidate, True)  if a valid candidate is found
      (candidate, False) if a candidate is found but invalid
      (None, False)      if no candidate list is found
    """
    original_nums = params["nums"]
    target = params.get("target", params.get("target_sum"))
    if target is None:
        raise KeyError('params must contain "target" (or "target_sum")')

    if k is None:
        k = params.get("k")
    if not isinstance(k, int) or k <= 0:
        raise ValueError('k must be a positive int (pass k=3/4 or set params["k"])')

    def is_valid(candidate: list[int]) -> bool:
        if not (isinstance(candidate, list) and all(isinstance(x, int) for x in candidate)):
            return False
        if len(candidate) != k:
            return False

        original_counter = Counter(original_nums)
        candidate_counter = Counter(candidate)
        # multiset containment: candidate elements must not exceed counts in nums
        if any(candidate_counter[num] > original_counter[num] for num in candidate_counter):
            return False

        return sum(candidate) == target

    # 1) Preferred: "Answer: [...]"
    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)
    if matches:
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if is_valid(candidate):
                return candidate, True
            # parsed but invalid
            if isinstance(candidate, list):
                return candidate, False
        except Exception:
            pass

    # 2) Fallback: any bracketed list in the text
    fallback_matches = re.findall(r'\[[^\]]*\]', text)
    for candidate_str in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(candidate_str)
            if is_valid(candidate):
                return candidate, True
            if isinstance(candidate, list):
                return candidate, False
        except Exception:
            continue

    return None, False

def extract_two_sum_from_text(text, params):
    return extract_k_sum_from_text(text, params, k=2)

def extract_three_sum_from_text(text, params):
    return extract_k_sum_from_text(text, params, k=3)

def extract_four_sum_from_text(text, params):
    return extract_k_sum_from_text(text, params, k=4)

def extract_four_sum_closest_with_exact_from_text(text, params):
    return extract_k_sum_from_text(text, params, k=4)

def extract_max_flow_from_text(text, params):
    """
    Extracts a candidate maximum element from text in a robust manner.
    It first checks for the expected format "Answer:" followed by a number.
    If found, it parses and validates whether it equals max(original_nums).
    If not found, it falls back to scanning the text.
    
    Returns:
      (candidate, True) if the candidate equals max(original_nums);
      (candidate, False) if a candidate is found but does not match;
      (None, False) if no candidate is found.
    """
    # First, try the expected format "Answer:" followed by a number.
    pattern = r'Answer:\s*(\d+)'
    answer = params['answer']
    matches = re.findall(pattern, text)
    if matches:
        # Take the last match as the final answer
        try:
            candidate = int(matches[-1])
            if candidate == answer:
                return candidate, True
            else:
                return candidate, False
        except Exception:
            pass

    # Fallback: search the text for any number.
    fallback_matches = re.findall(r'(\d+)', text)
    for m in reversed(fallback_matches):
        try:
            candidate = int(m)
            if candidate == answer:
                return candidate, True
            else:
                return candidate, False
        except Exception:
            continue
    return None, False

def _extract_int_sequence(bracket_content: str):
    """
    Extract integers in order from arbitrary content.
    Separator-agnostic: commas, arrows, text, emojis, spaces all allowed.
    """
    nums = re.findall(r'[-+]?\d+', bracket_content)
    if not nums:
        return None
    return [int(x) for x in nums]


def extract_shortest_path_from_text(text, params):
    """
    Extracts a candidate shortest path from text.

    Accepts ANY separator inside [...].
    Example:
      [4,0,3,5]
      [4 -> 0 -> 3 -> 5]
      [4 → 0 → 3 → 5]
      [4 foo 0 bar 3 baz 5]
      [4|0|3|5]

    Returns:
      (candidate, True)  if candidate == params['answer']
      (candidate, False) if a candidate is found but != answer
      (None, False)      if no candidate is found
    """
    answer = params['answer']

    # 1) Prefer explicit "Answer: [...]"
    matches = re.findall(r'Answer:\s*\[([^\]]*)\]', text)
    if matches:
        candidate = _extract_int_sequence(matches[-1])
        if candidate is not None:
            return candidate, (candidate == answer)

    # 2) Fallback: any [...]
    fallback_matches = re.findall(r'\[([^\]]+)\]', text)
    for m in reversed(fallback_matches):
        candidate = _extract_int_sequence(m)
        if candidate is not None:
            return candidate, (candidate == answer)

    return None, False


def extract_subset_sum_from_text(text, params):
    """
    Extracts a candidate subset from text (e.g. "[1, 4, 5]") in a robust manner.
    It first looks for the expected format "Answer:" followed by a bracketed list.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a list of integers.
      - The candidate is a valid subset of original_nums (using multiset containment).
      - The candidate's sum equals the target.
    
    Returns:
      (candidate, True) if a valid candidate is found;
      (candidate, False) if a candidate is found but fails validation;
      (None, False) if no candidate is found.
    """
    # First, try the expected format "Answer:" followed by a bracketed list.
    original_nums = params["nums"]
    target = params["target_sum"]
    pattern = r'Answer:\s*(\[[^\]]+\])'
    matches = re.findall(pattern, text)
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                original_counter = Counter(original_nums)
                candidate_counter = Counter(candidate)
                valid = all(original_counter[num] >= candidate_counter[num] for num in candidate_counter)
                if valid and sum(candidate) == target:
                    return candidate, True
                else:
                    return candidate, False
        except Exception:
            pass

    # Fallback: search for any bracketed list in the text.
    fallback_matches = re.findall(r'\[([^\]]+)\]', text)
    for m in reversed(fallback_matches):
        try:
            candidate_str = "[" + m + "]"
            candidate = ast.literal_eval(candidate_str)
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                original_counter = Counter(original_nums)
                candidate_counter = Counter(candidate)
                valid = all(original_counter[num] >= candidate_counter[num] for num in candidate_counter)
                if valid and sum(candidate) == target:
                    return candidate, True
                else:
                    return candidate, False
        except Exception:
            continue
    return None, False

def extract_sorting_from_text(text, params):
    """
    Extracts a candidate sorted list from text (e.g. "[1, 2, 3]") in a robust manner.
    It first checks for the expected format "Sorted list:" followed by a bracketed list.
    If found, it parses and compares the candidate to the correctly sorted version of original_nums.
    If not found or invalid, it falls back to scanning the text.
    
    Returns:
      (candidate, True) if the candidate equals sorted(original_nums);
      (candidate, False) if a candidate is found but is not equal;
      (None, False) if no candidate is found.
    """
    original_nums = params['nums']
    # First, try the expected format "Sorted list:" followed by a bracketed list.
    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = _normalize_int_list(ast.literal_eval(candidate_str))
            if candidate is not None:
                if candidate == sorted(original_nums):
                    return candidate, True
                else:
                    return candidate, False
        except Exception:
            pass

    # Fallback: search for any bracketed list.
    last_candidate = None
    fallback_matches = re.findall(r'\[([^\]]+)\]', text)
    for m in reversed(fallback_matches):
        try:
            candidate_str = "[" + m + "]"
            candidate = _normalize_int_list(ast.literal_eval(candidate_str))
            if candidate is None:
                continue
            if candidate == sorted(original_nums):
                return candidate, True
            if last_candidate is None:
                last_candidate = candidate
        except Exception:
            continue
    if last_candidate is not None:
        return last_candidate, False
    return None, False

def extract_bubble_sort_from_text(text, params):
    """
    Extracts a candidate sorted list from text in a robust manner.
    It first checks for the expected format "Sorted list:" followed by a bracketed list.
    If found, it parses and compares the candidate to the correctly sorted version of original_nums.
    If not found or invalid, it falls back to scanning the text.
    
    Returns:
      (candidate, True) if the candidate equals sorted(original_nums);
      (candidate, False) if a candidate is found but is not equal;
      (None, False) if no candidate is found.
    """
    return extract_sorting_from_text(text, params)

def extract_merge_sort_from_text(text, params):
    """
    Extracts a candidate sorted list from text in a robust manner.
    It first checks for the expected format "Sorted list:" followed by a bracketed list.
    If found, it parses and compares the candidate to the correctly sorted version of original_nums.
    If not found or invalid, it falls back to scanning the text.
    
    Returns:
      (candidate, True) if the candidate equals sorted(original_nums);
      (candidate, False) if a candidate is found but is not equal;
      (None, False) if no candidate is found.
    """
    return extract_sorting_from_text(text, params)

def extract_array_max_from_text(text, original_nums):
    """
    Extracts a candidate maximum element from text in a robust manner.
    It first checks for the expected format "Answer:" followed by a number.
    If found, it parses and validates whether it equals max(original_nums).
    If not found, it falls back to scanning the text.
    
    Returns:
      (candidate, True) if the candidate equals max(original_nums);
      (candidate, False) if a candidate is found but does not match;
      (None, False) if no candidate is found.
    """
    # First, try the expected format "Answer:" followed by a number.
    pattern = r'Answer:\s*(\d+)'
    original_nums = original_nums["nums"]
    matches = re.findall(pattern, text)
    if matches:
        # Take the last match as the final answer
        try:
            candidate = int(matches[-1])
            if candidate == max(original_nums):
                return candidate, True
            else:
                return candidate, False
        except Exception:
            pass

    # Fallback: search the text for any number.
    fallback_matches = re.findall(r'(\d+)', text)
    for m in reversed(fallback_matches):
        try:
            candidate = int(m)
            if candidate == max(original_nums):
                return candidate, True
            else:
                return candidate, False
        except Exception:
            continue
    return None, False

def extract_lps_from_text(text, original_s):
    """
    Extracts a candidate longest palindrome substring from text in a robust manner.
    It first checks for the expected format "Answer:" followed by a quoted string.
    If found, it validates whether it's a palindrome and is a substring of original_s.
    If not found, it falls back to scanning the text.
    
    Returns:
      (candidate, True) if the candidate is a valid palindrome substring;
      (candidate, False) if a candidate is found but is invalid;
      (None, False) if no candidate is found.
    """
    def longestPalindrome(s):
        n = len(s)
        if n == 0:
            return ""
        start, maxLen = 0, 1
        for i in range(n):
            for j in range(2):
                low, high = i, i + j
                while low >= 0 and high < n and s[low] == s[high]:
                    currLen = high - low + 1
                    if currLen > maxLen:
                        start = low
                        maxLen = currLen
                    low -= 1
                    high += 1
        return s[start:start + maxLen]

    def is_palindrome(s):
        return s == s[::-1]
    
    def is_substring(s, original):
        return s in original

    correct_lps = longestPalindrome(original_s)
    lps_length = len(correct_lps)

    # First, try the expected format "Answer:" followed by a quoted or raw string
    answer_lines = re.findall(r'(?im)^Answer:\s*(.+)$', text)
    if answer_lines:
        raw = answer_lines[-1].strip()
        candidate = None
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, str):
                candidate = parsed
        except Exception:
            pass
        if candidate is None:
            candidate = raw
            if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
                candidate = candidate[1:-1]
        if is_palindrome(candidate) and is_substring(candidate, original_s) and len(candidate) == lps_length:
            return candidate, True
        else:
            return candidate, False

    # Fallback: search for any quoted string
    last_candidate = None
    fallback_matches = re.findall(r'(["\'])(.*?)\1', text, flags=re.DOTALL)
    for _, candidate in reversed(fallback_matches):
        if is_palindrome(candidate) and is_substring(candidate, original_s) and len(candidate) == lps_length:
            return candidate, True
        if last_candidate is None:
            last_candidate = candidate
    if last_candidate is not None:
        return last_candidate, False
    return None, False

def extract_lis_from_text_origin(text, original_nums):
    """
    Extracts a candidate longest increasing subsequence from text in a robust manner.
    It first checks for the expected format "Answer:" followed by a bracketed list.
    If found, it validates whether it's a valid increasing subsequence of original_nums.
    If not found, it falls back to scanning the text.
    
    Returns:
      (candidate, True) if the candidate is a valid increasing subsequence;
      (candidate, False) if a candidate is found but is invalid;
      (None, False) if no candidate is found.
    """

    def is_increasing(nums):
        return all(nums[i] < nums[i+1] for i in range(len(nums)-1))
    
    def is_subsequence(candidate, original):
        if not candidate:
            return True
        i = 0
        for num in original:
            if i < len(candidate) and num == candidate[i]:
                i += 1
        return i == len(candidate)

    def lis(arr):
        n = len(arr)
        lis = [1] * n
        for i in range(1, n):
            for prev in range(0, i):
                if arr[i] > arr[prev]:
                    lis[i] = max(lis[i], lis[prev] + 1)
        return max(lis)
    
    lis_length = lis(original_nums)

    pattern = r'Answer:\s*(\[[^\]]+\])'
    matches = re.findall(pattern, text)
    if matches:
        try:
            candidate = ast.literal_eval(matches[-1])
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                if is_increasing(candidate) and is_subsequence(candidate, original_nums) and len(candidate) == lis_length:
                    return candidate, True
                else:
                    return candidate, False
        except Exception:
            pass

    # Fallback: search for any bracketed list
    fallback_matches = re.findall(r'\[([^\]]+)\]', text)
    for m in reversed(fallback_matches):
        try:
            candidate_str = "[" + m + "]"
            candidate = ast.literal_eval(candidate_str)
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                if is_increasing(candidate) and is_subsequence(candidate, original_nums) and len(candidate) == lis_length:
                    return candidate, True
                else:
                    return candidate, False
        except Exception:
            continue
    return None, False

def extract_lis_from_text(text, original_nums):
    """
    Extracts a candidate longest increasing subsequence from text in a robust manner.
    It first checks for the expected format "Answer:" followed by a bracketed list.
    If found, it validates whether it's a valid increasing subsequence of original_nums['nums'].
    If not found, it falls back to scanning the text.

    Returns:
      (candidate, True)  if the candidate is a valid LIS (strictly increasing subsequence of nums)
      (candidate, False) if a candidate is found but invalid
      (None, False)      if no candidate is found
    """

    # ---- normalize inputs: accept dict or list ----
    if isinstance(original_nums, dict):
        nums_list = original_nums.get('nums', [])
        target = original_nums.get('target', None)
    else:
        nums_list = list(original_nums)
        target = None

    # ---- helpers ----
    def is_increasing(nums):
        return all(nums[i] < nums[i+1] for i in range(len(nums)-1))

    def is_subsequence(candidate, original):
        if not candidate:
            return True
        i = 0
        for num in original:
            if i < len(candidate) and num == candidate[i]:
                i += 1
        return i == len(candidate)

    def lis_length(arr):
        n = len(arr)
        if n == 0:
            return 0
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if arr[i] > arr[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    # If target is provided, use its length; LIS can be non-unique.
    if target is not None:
        required_len = len(target)
    else:
        required_len = lis_length(nums_list)

    # ---- try to parse "Answer: [ ... ]" first ----
    pattern = r'Answer:\s*(\[[^\]]+\])'
    matches = re.findall(pattern, text)
    if matches:
        try:
            candidate = ast.literal_eval(matches[-1])
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                valid = is_increasing(candidate) and is_subsequence(candidate, nums_list) and (len(candidate) == required_len)
                return candidate, bool(valid)
        except Exception:
            # fall through to generic scan
            pass

    # ---- fallback: scan any bracketed list ----
    fallback_matches = re.findall(r'\[([^\]]+)\]', text)
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval("[" + m + "]")
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                valid = is_increasing(candidate) and is_subsequence(candidate, nums_list) and (len(candidate) == required_len)
                return candidate, bool(valid)
        except Exception:
            continue

    return None, False

def extract_merged_intervals_from_text(text, original_intervals):
    """
    Extracts and validates the merged intervals result from model output.
    - Looks for "Merged:" followed by a list of lists.
    - Falls back to finding any list of lists.
    - Validates that it is a list of two-item lists with start <= end.
    """
    def merge_intervals(intervals):
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
    
    def try_parse(candidate_str):
        try:
            candidate = ast.literal_eval(candidate_str)
            if isinstance(candidate, list) and all(
                isinstance(interval, list) and len(interval) == 2 and interval[0] <= interval[1]
                for interval in candidate
            ):
                return candidate
        except Exception:
            pass
        return None

    correct = merge_intervals(original_intervals)

    pattern = r'Answer:\s*(\[\[.*?\]\])'
    matches = re.findall(pattern, text)
    if matches:
        candidate = try_parse(matches[-1])
        if candidate is not None:
            return candidate, sorted(candidate) == correct
    
    fallback_matches = re.findall(r'(\[\[.*?\]\])', text)
    for match in reversed(fallback_matches):
        candidate = try_parse(match)
        if candidate is not None:
            return candidate, sorted(candidate) == correct

    return None, False

def extract_duplicates_from_text(text, original_nums):
    """
    Extracts and validates a list of duplicate numbers from the model output.
    - Looks for format like "Duplicates: [2, 3]"
    - Falls back to any bracketed list if needed
    - Validates that each number appears more than once in original_nums
    """
    pattern = r'Answer:\s*(\[[^\]]+\])'
    matches = re.findall(pattern, text)
    if matches:
        try:
            candidate = ast.literal_eval(matches[-1])
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                counter = Counter(original_nums)
                expected = sorted([num for num, count in counter.items() if count > 1])
                return sorted(candidate), sorted(candidate) == expected
        except Exception:
            pass

    # Fallback
    fallback_matches = re.findall(r'\[([^\]]+)\]', text)
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(f"[{m}]")
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                counter = Counter(original_nums)
                expected = sorted([num for num, count in counter.items() if count > 1])
                return sorted(candidate), sorted(candidate) == expected
        except Exception:
            continue
    return None, False

def extract_binary_search_from_text(text, params):
    """
    Extracts and validates the index returned by the model.
    - Searches for 'Answer: X'
    - Validates that the returned index matches the dataset ground truth
    """
    nums = params['nums']
    target_idx = params.get('target')
    target_val = params.get('target_val')

    if target_val is None and target_idx is not None and 0 <= target_idx < len(nums):
        target_val = nums[target_idx]

    def is_correct_index(index):
        if not (0 <= index < len(nums)):
            return False
        if target_idx is not None and index != target_idx:
            return False
        if target_val is not None and nums[index] != target_val:
            return False
        return True

    index = _extract_last_intlike_candidate(text)
    if index is None:
        return None, False
    return index, is_correct_index(index)

def extract_sliding_window_max_from_text(text, params):
    nums = params["nums"]
    k = params["k"]
    gt = params["answer"]

    def is_valid(cand):
        return isinstance(cand, list) and cand == gt

    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)

    if matches:
        try:
            cand = ast.literal_eval(matches[-1])
            return cand, is_valid(cand)
        except:
            pass

    return None, False

def extract_gcd_from_text(text, a, b):
    """
    Extracts the integer answer after 'Answer:' and validates it is the correct GCD of a and b.
    """
    candidate = _extract_last_intlike_candidate(text)
    if candidate is None:
        return None, False

    from math import gcd
    correct_gcd = gcd(a, b)
    return candidate, candidate == correct_gcd

def extract_regex_match_from_text(text, s, p):
    """
    Extracts 'Answer: True/False' and verifies it against Python's regex matching logic using dynamic programming.
    """

    def isMatch(t: str, p: str) -> bool:
        n = len(t)
        m = len(p)

        dp = [[False] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for j in range(1, m + 1):
            if p[j - 1] == '*' and j > 1:
                dp[0][j] = dp[0][j - 2]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if p[j - 1] == '.' or t[i - 1] == p[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*' and j > 1:
                    dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (p[j - 2] == t[i - 1] or p[j - 2] == '.'))

        return dp[n][m]

    pattern = r'Answer:\s*(true|false)'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    ground_truth = isMatch(s, p)

    if matches:
        answer_str = matches[-1].lower()
        candidate = answer_str == "true"
        return candidate, candidate == ground_truth
    
    # Fallback: search for any boolean
    fallback_matches = re.findall(r'\b(true|false)\b', text, flags=re.IGNORECASE)
    for m in reversed(fallback_matches):
        answer_str = m.lower()
        candidate = answer_str == "true"
        return candidate, candidate == ground_truth
    # If no match found, return None

    return None, False

def extract_sat_from_text(text, params):
    """
    Extracts the answer of the form:
    Answer: {x1=true, x2=false, ...}
    OR: Answer: UNSATISFIABLE

    Validates that the assignment satisfies all clauses.
    """
    cnf_clauses = params['cnf_clauses']
    var_list = params['var_list']

    def is_cnf_satisfied(cnf_clauses, assignment):
        """
        Checks if a given Boolean assignment satisfies all CNF clauses.
        `assignment` is a list of booleans, index i corresponds to xi+1.
        """
        for clause in cnf_clauses:
            clause_satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                val = assignment[var_idx]
                if (lit > 0 and val) or (lit < 0 and not val):
                    clause_satisfied = True
                    break
            if not clause_satisfied:
                return False
        return True
    
    pattern_unsat = r'Answer:\s*UNSATISFIABLE'
    if re.search(pattern_unsat, text, re.IGNORECASE):
        # Brute-force verify no satisfying assignment exists
        from itertools import product
        all_assignments = list(product([False, True], repeat=len(var_list)))
        for assignment in all_assignments:
            if is_cnf_satisfied(cnf_clauses, assignment):
                return "UNSATISFIABLE", False  # model was wrong
        return "UNSATISFIABLE", True

    # Otherwise look for an assignment dict
    pattern_assign = r'Answer:\s*\{([^\}]+)\}'
    matches = re.findall(pattern_assign, text)
    if matches:
        try:
            items = matches[-1].split(",")
            assignment = {}
            for item in items:
                var, val = item.strip().split("=")
                var = var.strip().lower()
                val = val.strip().lower()
                assignment[var] = val == "true"

            # Convert to list in order
            ordered_assignment = [assignment.get(f"x{i+1}", False) for i in range(len(var_list))]
            satisfied = is_cnf_satisfied(cnf_clauses, ordered_assignment)
            return assignment, satisfied
        except Exception:
            pass
    return None, False

def extract_knapsack_selection_from_text(text, items, capacity):
    """
    Extracts a selected list of items and validates:
    - All selected items are from the original item list
    - Total weight is ≤ capacity
    - Total value is the optimal one
    """
    def solve_knapsack_optimal_value(items, capacity):
        """
        Standard 0/1 Knapsack DP to compute maximum achievable value.
        """
        n = len(items)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(n):
            w, v = items[i]
            for c in range(capacity + 1):
                if w <= c:
                    dp[i + 1][c] = max(dp[i][c], dp[i][c - w] + v)
                else:
                    dp[i + 1][c] = dp[i][c]

        return dp[n][capacity]

    pattern = r'Answer:\s*(\[\([^\]]*\)\])'
    matches = re.findall(pattern, text)
    if matches:
        try:
            selected_items = ast.literal_eval(matches[-1])
            if isinstance(selected_items, list) and all(isinstance(t, tuple) and len(t) == 2 for t in selected_items):
                # Validate items exist in original list (multiset containment)
                original_counter = Counter(items)
                selected_counter = Counter(selected_items)
                valid_items = all(original_counter[i] >= selected_counter[i] for i in selected_counter)

                total_weight = sum(w for w, _ in selected_items)
                total_value = sum(v for _, v in selected_items)

                # Compute optimal solution using DP
                best_value = solve_knapsack_optimal_value(items, capacity)

                is_correct = valid_items and total_weight <= capacity and total_value == best_value
                return selected_items, is_correct
        except Exception:
            pass

    return None, False

def extract_longest_valid_parentheses_from_text(text, s):
    """
    Extracts the integer answer after 'Answer:' and checks correctness by computing the true max length.
    """

    def solve_longest_valid_parentheses(s):
        maxLen = 0
        open = close = 0
        for ch in s:
            if ch == '(':
                open += 1
            elif ch == ')':
                close += 1
            if open == close:
                maxLen = max(maxLen, 2 * close)
            elif close > open:
                open = close = 0
        open = close = 0
        for ch in reversed(s):
            if ch == '(':
                open += 1
            elif ch == ')':
                close += 1
            if open == close:
                maxLen = max(maxLen, 2 * open)
            elif open > close:
                open = close = 0
        return maxLen

    predicted = _extract_last_intlike_candidate(text)
    if predicted is None:
        return None, False

    actual = solve_longest_valid_parentheses(s)
    return predicted, predicted == actual


def extract_vector_mean_from_text(text, params):
    return extract_vector_mean_calculation_from_text(text, params)

def extract_kth_permutation_from_text(text, n, k):
    """
    Extracts the output string after 'Answer:' and validates it against the true kth permutation.
    """
    def permutation(n, k, nums, result_str):
        # precalculated factorials
        fact = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800]

        if n == 0:
            return result_str
        if k <= 1 or k <= fact[n - 1]:
            val = nums[-1] if k == 0 else nums[0]
        else:
            index = k // fact[n - 1]
            k = k % fact[n - 1]  # remaining permutations
            if k == 0:
                index -= 1
            val = nums[index]
        result_str += str(val)
        nums.remove(val)
        return permutation(n - 1, k, nums, result_str)

    def get_permutation(n, k):
        nums = list(range(1, n + 1))
        result_str = ""
        return permutation(n, k, nums, result_str)

    pattern = r'Answer:\s*"([1-9]+)"'
    matches = re.findall(pattern, text)
    if matches:
        predicted = matches[-1]
        actual = get_permutation(n, k)
        return predicted, predicted == actual
    
    # Fallback: search for any string of digits
    fallback_matches = re.findall(r'"([1-9]+)"', text)
    for m in reversed(fallback_matches):
        predicted = m
        actual = get_permutation(n, k)
        return predicted, predicted == actual
    
    return None, False

def extract_3d_longest_common_substring_from_text(text, sequence_x, sequence_y, sequence_z):
    """
    Extracts a candidate longest common substring from text in a robust manner.
    It looks for the expected format "Answer: "..." (length N)".
    If found, it parses and validates the candidate.
    
    Validation ensures:
      - The substring is present in all three input sequences
      - The length matches the claimed length
      - It is indeed the longest common substring
    
    Returns:
      (substring, length, True) if a valid answer is found;
      (substring, length, False) if an answer is found but fails validation;
      (None, None, False) if no answer is found.
    """
    import re
    
    # Helper function to find the longest common substring of three strings
    def find_longest_common_substring(s1, s2, s3):
        len1, len2, len3 = len(s1), len(s2), len(s3)
        
        # Initialize 3D DP table
        dp = [[[0 for _ in range(len3+1)] for _ in range(len2+1)] for _ in range(len1+1)]
        
        # Variables to store the maximum length and ending position
        max_length = 0
        end_pos = (0, 0, 0)
        
        # Fill the DP table
        for i in range(1, len1+1):
            for j in range(1, len2+1):
                for k in range(1, len3+1):
                    if s1[i-1] == s2[j-1] == s3[k-1]:
                        dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                        if dp[i][j][k] > max_length:
                            max_length = dp[i][j][k]
                            end_pos = (i, j, k)
        
        # Extract the substring
        i, j, k = end_pos
        if max_length == 0:
            return ""
        
        # Reconstruct the substring using the ending position
        return s1[i-max_length:i]
    
    # First, try the expected format "Answer: "..." (length N)"
    pattern = r'Answer:\s*"([^"]*)"\s*\(length\s*(\d+)\)'
    matches = re.findall(pattern, text)
    
    if not matches:
        # Try with single quotes
        pattern = r"Answer:\s*'([^']*)'\s*\(length\s*(\d+)\)"
        matches = re.findall(pattern, text)
    
    if matches:
        # Take the last match as the final answer
        substring, length_str = matches[-1]
        claimed_length = int(length_str)
        
        # Validate the substring
        is_in_x = substring in sequence_x
        is_in_y = substring in sequence_y
        is_in_z = substring in sequence_z
        
        # Check if it's valid
        if is_in_x and is_in_y and is_in_z and len(substring) == claimed_length:
            # Verify it's the longest
            actual_lcs = find_longest_common_substring(sequence_x, sequence_y, sequence_z)
            if len(substring) == len(actual_lcs):
                return substring, claimed_length, True
        
        return substring, claimed_length, False
    
    # Fallback: look for any quoted string and length
    substring_pattern = r'"([^"]*)"'
    length_pattern = r'length\s*(?:is|=)?\s*(\d+)'
    
    substring_matches = re.findall(substring_pattern, text)
    if not substring_matches:
        substring_pattern = r"'([^']*)'"
        substring_matches = re.findall(substring_pattern, text)
    
    length_matches = re.findall(length_pattern, text)
    
    if substring_matches and length_matches:
        substring = substring_matches[-1]
        claimed_length = int(length_matches[-1])
        
        # Validate the substring
        is_in_x = substring in sequence_x
        is_in_y = substring in sequence_y
        is_in_z = substring in sequence_z
        
        if is_in_x and is_in_y and is_in_z and len(substring) == claimed_length:
            actual_lcs = find_longest_common_substring(sequence_x, sequence_y, sequence_z)
            if len(substring) == len(actual_lcs):
                return substring, claimed_length, True
        
        return substring, claimed_length, False
    
    return None, None, False

def extract_shortest_path_from_text_old(text, graph, source, target):
    """
    Extracts a candidate shortest path result from text in a robust manner.
    It looks for the expected format "Answer: Distance = X (path: u→v→...→w)".
    If found, it parses and validates the candidate.
    
    Validation ensures:
      - The path exists in the graph
      - The path starts at source and ends at target
      - The total distance matches the claimed distance
    
    Returns:
      (distance, path, True) if a valid answer is found;
      (distance, path, False) if an answer is found but fails validation;
      (None, None, False) if no answer is found.
    """
    import re
    
    # Convert graph to adjacency list for easier validation
    adj_list = {}
    for u, v, weight in graph:
        if u not in adj_list:
            adj_list[u] = []
        adj_list[u].append((v, weight))
    
    # Helper function to validate a path and compute its distance
    def validate_path(path):
        if not path or len(path) < 2:
            return False, None
        
        # Check if path starts at source and ends at target
        if path[0] != source or path[-1] != target:
            return False, None
        
        # Compute total distance and check if edges exist
        total_distance = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u not in adj_list:
                return False, None
            
            # Find weight of edge (u, v)
            edge_found = False
            for neighbor, weight in adj_list[u]:
                if neighbor == v:
                    total_distance += weight
                    edge_found = True
                    break
            
            if not edge_found:
                return False, None
        
        return True, total_distance
    
    # First, try the expected format "Answer: Distance = X (path: u→v→...→w)"
    pattern = r'Answer:\s*Distance\s*=\s*(-?\d+)\s*\(path:\s*([^)]+)\)'
    matches = re.findall(pattern, text)
    
    if matches:
        # Take the last match as the final answer
        claimed_distance_str, path_str = matches[-1]
        try:
            claimed_distance = int(claimed_distance_str)
            
            # Parse the path, handling different arrow formats (→, ->, etc.)
            path_str = path_str.replace('→', '->').replace('-->', '->').replace(' -> ', '->').replace(' ->', '->').replace('-> ', '->')
            path = [int(v) for v in path_str.split('->')]
            
            is_valid, actual_distance = validate_path(path)
            if is_valid and actual_distance == claimed_distance:
                return claimed_distance, path, True
            else:
                return claimed_distance, path, False
        except Exception as e:
            pass
    
    # Fallback: look for any distance and path-like format
    distance_pattern = r'[Dd]istance\s*(?:is|=)\s*(-?\d+)'
    path_pattern = r'[Pp]ath(?:\s*is|:)\s*([0-9→\->,\s]+)'
    
    distance_matches = re.findall(distance_pattern, text)
    path_matches = re.findall(path_pattern, text)
    
    if distance_matches and path_matches:
        try:
            claimed_distance = int(distance_matches[-1])
            
            path_str = path_matches[-1].replace('→', '->').replace('-->', '->').replace(' -> ', '->').replace(' ->', '->').replace('-> ', '->')
            path = [int(v) for v in re.findall(r'\d+', path_str)]
            
            is_valid, actual_distance = validate_path(path)
            if is_valid and actual_distance == claimed_distance:
                return claimed_distance, path, True
            else:
                return claimed_distance, path, False
        except Exception:
            pass
    
    return None, None, False

def extract_longest_common_substring_from_text(text, string1, string2):
    """
    Extracts a candidate longest common substring from text (e.g. "Answer: "hello"") in a robust manner.
    It first looks for the expected format "Answer:" followed by a quoted string.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a string.
      - The candidate is a substring of both string1 and string2.
      - The candidate has length > 0.
    
    Returns:
      (candidate, True) if a valid candidate is found;
      (candidate, False) if a candidate is found but fails validation;
      (None, False) if no candidate is found.
    """
    # First, try the expected format "Answer:" followed by a quoted string.
    pattern = r'Answer:\s*"([^"]*)"'
    matches = re.findall(pattern, text)
    if not matches:
        # Try with single quotes
        pattern = r"Answer:\s*'([^']*)'"
        matches = re.findall(pattern, text)
    
    if matches:
        # Take the last match as the final answer
        candidate = matches[-1]
        if candidate in string1 and candidate in string2 and len(candidate) > 0:
            return candidate, True
        else:
            return candidate, False
    
    # Fallback: search for any quoted string in the text.
    fallback_matches = re.findall(r'"([^"]*)"', text)
    if not fallback_matches:
        fallback_matches = re.findall(r"'([^']*)'", text)
        
    for candidate in reversed(fallback_matches):
        if candidate in string1 and candidate in string2 and len(candidate) > 0:
            return candidate, True
        else:
            return candidate, False
    
    return None, False

# def extract_scalar_vector_multiplication_from_text(text, scalar, vector):
def extract_scalar_vector_multiplication_from_text(text, params):
    """
    Extracts a candidate scalar-vector multiplication result from text in a robust manner.
    It first looks for the expected format "Answer:" followed by a list.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a valid vector (list of numbers).
      - The length matches the input vector.
      - The values match the expected result of multiplying scalar and vector.
    
    Returns:
      (candidate, True) if a valid candidate is found;
      (candidate, False) if a candidate is found but fails validation;
      (None, False) if no candidate is found.
    """
    scalar = params['scalar']
    vector = params['vector']
    
    import re
    import ast
    import numpy as np
    
    # Helper function to validate vector dimensions
    def validate_dimensions(result, v):
        if not result or not isinstance(result, list):
            return False
        return len(result) == len(v)
    
    # Helper function to compute expected result for validation
    def compute_expected(s, v):
        try:
            return [s * x for x in v]
        except:
            return None
    
    # First, try the expected format "Answer:" followed by a list
    pattern = r'Answer:\s*(\[.+?\])'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if validate_dimensions(candidate, vector):
                expected = compute_expected(scalar, vector)
                if expected is not None:
                    # Check if values are approximately equal (for floating point)
                    candidate_np = np.array(candidate)
                    expected_np = np.array(expected)
                    if np.allclose(candidate_np, expected_np, rtol=1e-5, atol=1e-8):
                        return candidate, True
            return candidate, False
        except Exception:
            pass
    
    # Fallback: search for any list structure in the text
    fallback_pattern = r'\[(?:[^\[\]]+)\]'
    fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
    
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(m)
            if validate_dimensions(candidate, vector):
                expected = compute_expected(scalar, vector)
                if expected is not None:
                    candidate_np = np.array(candidate)
                    expected_np = np.array(expected)
                    if np.allclose(candidate_np, expected_np, rtol=1e-5, atol=1e-8):
                        return candidate, True
            return candidate, False
        except Exception:
            continue
    
    return None, False

# def extract_matrix_vector_multiplication_from_text(text, matrix, vector):
def extract_matrix_vector_multiplication_from_text(text, params):
    """
    Extracts a candidate matrix-vector multiplication result from text in a robust manner.
    It first looks for the expected format "Answer:" followed by a list.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a valid vector (list of numbers).
      - The dimensions are correct for matrix-vector multiplication (matrix rows).
      - The values match the expected result of multiplying the matrix and vector.
    
    Returns:
      (candidate, True) if a valid candidate is found;
      (candidate, False) if a candidate is found but fails validation;
      (None, False) if no candidate is found.
    """
    matrix = params['matrix']
    vector = params['vector']
    
    import re
    import ast
    import numpy as np
    
    # Helper function to validate vector dimensions
    def validate_dimensions(result, m):
        if not result or not isinstance(result, list):
            return False
        expected_length = len(m)  # Number of rows in the matrix
        return len(result) == expected_length
    
    # Helper function to compute expected result for validation
    def compute_expected(m, v):
        try:
            m_np = np.array(m)
            v_np = np.array(v)
            return m_np.dot(v_np).tolist()
        except:
            return None
    
    # First, try the expected format "Answer:" followed by a list
    pattern = r'Answer:\s*(\[.+?\])'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if validate_dimensions(candidate, matrix):
                expected = compute_expected(matrix, vector)
                if expected is not None:
                    # Check if values are approximately equal (for floating point)
                    candidate_np = np.array(candidate)
                    expected_np = np.array(expected)
                    if np.allclose(candidate_np, expected_np, rtol=1e-5, atol=1e-8):
                        return candidate, True
            return candidate, False
        except Exception:
            pass
    
    # Fallback: search for any list structure in the text
    fallback_pattern = r'\[(?:[^\[\]]+)\]'
    fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
    
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(m)
            if validate_dimensions(candidate, matrix):
                expected = compute_expected(matrix, vector)
                if expected is not None:
                    candidate_np = np.array(candidate)
                    expected_np = np.array(expected)
                    if np.allclose(candidate_np, expected_np, rtol=1e-5, atol=1e-8):
                        return candidate, True
            return candidate, False
        except Exception:
            continue
    
    return None, False

# def extract_matrix_multiplication_from_text(text, matrix_a, matrix_b):
def extract_matrix_multiplication_from_text(text, params):
    """
    Extracts a candidate matrix multiplication result from text in a robust manner.
    It first looks for the expected format "Answer:" followed by a nested list.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a valid matrix (list of lists of numbers).
      - The dimensions are correct for matrix multiplication (A's rows × B's columns).
      - The values match the expected result of multiplying A and B.
    
    Returns:
      (candidate, True) if a valid candidate is found;
      (candidate, False) if a candidate is found but fails validation;
      (None, False) if no candidate is found.
    """
    matrix_a = params['matrix_a']
    matrix_b = params['matrix_b']
    
    import re
    import ast
    import numpy as np
    
    # Helper function to validate matrix dimensions
    def validate_dimensions(result, a, b):
        if not result or not isinstance(result, list) or not all(isinstance(row, list) for row in result):
            return False
        expected_rows = len(a)
        expected_cols = len(b[0]) if b and b[0] else 0
        return len(result) == expected_rows and all(len(row) == expected_cols for row in result)
    
    # Helper function to compute expected result for validation
    def compute_expected(a, b):
        a_np = np.array(a)
        b_np = np.array(b)
        try:
            return a_np.dot(b_np).tolist()
        except:
            return None
    
    # First, try the expected format "Answer:" followed by a nested list
    pattern = r'Answer:\s*(\[\s*\[.+?\]\s*\])'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if validate_dimensions(candidate, matrix_a, matrix_b):
                expected = compute_expected(matrix_a, matrix_b)
                if expected is not None:
                    # Check if values are approximately equal (for floating point)
                    candidate_np = np.array(candidate)
                    expected_np = np.array(expected)
                    if np.allclose(candidate_np, expected_np, rtol=1e-5, atol=1e-8):
                        return candidate, True
            return candidate, False
        except Exception:
            pass
    
    # Fallback: search for any nested list structure in the text
    fallback_pattern = r'\[\s*\[(?:[^\[\]]+)\](?:\s*,\s*\[(?:[^\[\]]+)\])*\s*\]'
    fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
    
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(m)
            if validate_dimensions(candidate, matrix_a, matrix_b):
                expected = compute_expected(matrix_a, matrix_b)
                if expected is not None:
                    candidate_np = np.array(candidate)
                    expected_np = np.array(expected)
                    if np.allclose(candidate_np, expected_np, rtol=1e-5, atol=1e-8):
                        return candidate, True
            return candidate, False
        except Exception:
            continue
    
    return None, False

# def extract_vector_mean_calculation_from_text(text, data, precision_levels=None):
def extract_vector_mean_calculation_from_text(text, params, precision_levels=None):
    """
    Extracts a candidate vector mean calculation result from text in a robust manner.
    It first looks for the expected format "Answer:" followed by a list.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a valid vector (list of numbers).
      - The dimensions match the original data vectors.
      - The values match the expected result of calculating the element-wise mean,
        with configurable precision levels for flexibility.
    
    Parameters:
    - text: The text containing the potential answer
    - data: The original dataset (list of vectors)
    - precision_levels: A dictionary specifying different precision levels for validation:
        - 'exact': For bit-perfect answers (default: rtol=1e-10, atol=1e-10)
        - 'high': For high precision answers (default: rtol=1e-5, atol=1e-8)
        - 'medium': For medium precision answers (default: rtol=1e-3, atol=1e-5)
        - 'low': For low precision answers (default: rtol=1e-2, atol=1e-3)
        - 'rough': For approximate answers (default: rtol=1e-1, atol=1e-2)
    
    Returns:
      (candidate, True, precision_level) if a valid candidate is found;
      (candidate, False, None) if a candidate is found but fails validation;
      (None, False, None) if no candidate is found.
    """
    data = params['data']
    #if |a - b| ≤ atol + rtol * |b|, then a and b are viewed as the same(correct answer)
    # Define default precision levels if not provided
    if precision_levels is None:
        precision_levels = {
            'exact': {'rtol': 1e-10, 'atol': 1e-10},   # Bit-perfect
            'high': {'rtol': 1e-5, 'atol': 1e-8},      # High precision
            'medium': {'rtol': 1e-3, 'atol': 1e-5},    # Medium precision
            'low': {'rtol': 1e-2, 'atol': 1e-3},       # Low precision
            'rough': {'rtol': 1e-1, 'atol': 1e-2}      # Rough approximation
        }
    
    # Helper function to validate vector dimensions
    def validate_dimensions(result, data_list):
        if not result or not isinstance(result, list):
            return False
        if not data_list or not data_list[0]:
            return False
        expected_dim = len(data_list[0])
        return len(result) == expected_dim
    
    # Helper function to compute expected result for validation
    def compute_expected(data_list):
        try:
            if params.get("target") is not None:
                return params["target"]
            if not data_list or not data_list[0]:
                return None

            n = len(data_list)
            d = len(data_list[0])
            return [sum(float(row[i]) for row in data_list) / n for i in range(d)]
        except:
            return None
    
    # Helper function to check if values match at a given precision level
    def check_precision(candidate_values, expected_values, precision_level):
        rtol = precision_levels[precision_level]['rtol']
        atol = precision_levels[precision_level]['atol']
        try:
            if len(candidate_values) != len(expected_values):
                return False
            for a, b in zip(candidate_values, expected_values):
                if abs(float(a) - float(b)) > atol + rtol * abs(float(b)):
                    return False
            return True
        except Exception:
            return False
    
    # First, try the expected format "Answer:" followed by a list
    pattern = r'Answer:\s*(\[.+?\])'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if validate_dimensions(candidate, data):
                expected = compute_expected(data)
                if expected is not None:
                    # Check at different precision levels, starting with the most precise
                    for level in ['exact', 'high', 'medium', 'low', 'rough']:
                        if check_precision(candidate, expected, level):
                            return candidate, True, level
            return candidate, False, None
        except Exception:
            pass
    
    # Fallback: search for any list structure in the text
    fallback_pattern = r'\[(?:[^\[\]]+)\]'
    fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
    
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(m)
            if validate_dimensions(candidate, data):
                expected = compute_expected(data)
                if expected is not None:
                    # Check at different precision levels
                    for level in ['exact', 'high', 'medium', 'low', 'rough']:
                        if check_precision(candidate, expected, level):
                            return candidate, True, level
            return candidate, False, None
        except Exception:
            continue
    
    return None, False, None

# def extract_covariance_matrix_from_text(text, data, precision_levels=None):
def extract_covariance_matrix_from_text(text, params, precision_levels=None):
    """
    Extracts a candidate covariance matrix calculation result from text in a robust manner.
    It first looks for the expected format "Answer:" followed by a nested list.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a valid matrix (list of lists of numbers).
      - The dimensions are correct for a covariance matrix (n×n where n is the dimension of data vectors).
      - The values match the expected result of calculating the covariance matrix,
        with configurable precision levels for flexibility.
      - The matrix is symmetric (within numerical precision).
    
    Parameters:
    - text: The text containing the potential answer
    - data: The original dataset (list of vectors)
    - precision_levels: A dictionary specifying different precision levels for validation:
        - 'exact': For bit-perfect answers (default: rtol=1e-10, atol=1e-10)
        - 'high': For high precision answers (default: rtol=1e-5, atol=1e-8)
        - 'medium': For medium precision answers (default: rtol=1e-3, atol=1e-5)
        - 'low': For low precision answers (default: rtol=1e-2, atol=1e-3)
        - 'rough': For approximate answers (default: rtol=1e-1, atol=1e-2)
    
    Returns:
      (candidate, True, precision_level) if a valid candidate is found;
      (candidate, False, None) if a candidate is found but fails validation;
      (None, False, None) if no candidate is found.
    """
    data = params['data']

    # Define default precision levels if not provided
    if precision_levels is None:
        precision_levels = {
            'exact': {'rtol': 1e-10, 'atol': 1e-10},   # Bit-perfect
            'high': {'rtol': 1e-5, 'atol': 1e-8},      # High precision
            'medium': {'rtol': 1e-3, 'atol': 1e-5},    # Medium precision
            'low': {'rtol': 1e-2, 'atol': 1e-3},       # Low precision
            'rough': {'rtol': 1e-1, 'atol': 1e-2}      # Rough approximation
        }
    
    # Helper function to validate matrix dimensions
    def validate_dimensions(result, data_list):
        if not result or not isinstance(result, list) or not all(isinstance(row, list) for row in result):
            return False
        if not data_list or not data_list[0]:
            return False
        
        # For covariance matrix, it should be square with dimensions equal to the number of features (vector length)
        feature_dim = len(data_list[0])
        return len(result) == feature_dim and all(len(row) == feature_dim for row in result)
    
    # Helper function to validate matrix symmetry
    def validate_symmetry(matrix, level='high'):
        rtol = precision_levels[level]['rtol']
        atol = precision_levels[level]['atol']
        try:
            n = len(matrix)
            for i in range(n):
                for j in range(n):
                    a = float(matrix[i][j])
                    b = float(matrix[j][i])
                    if abs(a - b) > atol + rtol * abs(b):
                        return False
            return True
        except Exception:
            return False
    
    # Helper function to compute expected result for validation
    def compute_expected(data_list):
        try:
            if params.get("target") is not None:
                return params["target"]
            if not data_list or not data_list[0]:
                return None

            n = len(data_list)
            d = len(data_list[0])
            means = [sum(float(row[i]) for row in data_list) / n for i in range(d)]
            cov = []
            for i in range(d):
                row = []
                for j in range(d):
                    num = 0.0
                    for sample in data_list:
                        num += (float(sample[i]) - means[i]) * (float(sample[j]) - means[j])
                    row.append(num / (n - 1))
                cov.append(row)
            return cov
        except:
            return None
    
    # Helper function to check if values match at a given precision level
    def check_precision(candidate_values, expected_values, precision_level):
        rtol = precision_levels[precision_level]['rtol']
        atol = precision_levels[precision_level]['atol']
        try:
            if len(candidate_values) != len(expected_values):
                return False
            for row_a, row_b in zip(candidate_values, expected_values):
                if len(row_a) != len(row_b):
                    return False
                for a, b in zip(row_a, row_b):
                    if abs(float(a) - float(b)) > atol + rtol * abs(float(b)):
                        return False
            return True
        except Exception:
            return False
    
    # First, try the expected format "Answer:" followed by a nested list
    pattern = r'Answer:\s*(\[\s*\[.+?\]\s*\])'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if validate_dimensions(candidate, data):
                expected = compute_expected(data)
                if expected is not None:
                    # Check if the matrix is symmetric first (a basic property of covariance matrices)
                    if not validate_symmetry(candidate, 'medium'):
                        return candidate, False, None
                    
                    # Check at different precision levels, starting with the most precise
                    for level in ['exact', 'high', 'medium', 'low', 'rough']:
                        if check_precision(candidate, expected, level):
                            return candidate, True, level
            return candidate, False, None
        except Exception:
            pass
    
    # Fallback: search for any nested list structure in the text
    fallback_pattern = r'\[\s*\[(?:[^\[\]]+)\](?:\s*,\s*\[(?:[^\[\]]+)\])*\s*\]'
    fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
    
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(m)
            if validate_dimensions(candidate, data):
                expected = compute_expected(data)
                if expected is not None:
                    # Check symmetry
                    if not validate_symmetry(candidate, 'medium'):
                        return candidate, False, None
                    
                    # Check at different precision levels
                    for level in ['exact', 'high', 'medium', 'low', 'rough']:
                        if check_precision(candidate, expected, level):
                            return candidate, True, level
            return candidate, False, None
        except Exception:
            continue
    
    return None, False, None

# def extract_inverse_covariance_matrix_from_text(text, data, precision_levels=None):
def extract_inverse_covariance_matrix_from_text(text, params, precision_levels=None):
    """
    Extracts a candidate inverse covariance matrix calculation from text in a robust manner.
    It first looks for the expected format "Answer:" followed by a nested list.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Validation ensures:
      - The candidate is a valid matrix (list of lists of numbers).
      - The dimensions are correct for an inverse covariance matrix (n×n where n is the dimension of data vectors).
      - The values match the expected result of calculating the inverse covariance matrix,
        with configurable precision levels for flexibility.
      - The matrix is symmetric (within numerical precision).
      - The product of the covariance matrix and its candidate inverse is approximately the identity matrix.
    
    Parameters:
    - text: The text containing the potential answer
    - data: The original dataset (list of vectors)
    - precision_levels: A dictionary specifying different precision levels for validation:
        - 'exact': For bit-perfect answers (default: rtol=1e-10, atol=1e-10)
        - 'high': For high precision answers (default: rtol=1e-5, atol=1e-8)
        - 'medium': For medium precision answers (default: rtol=1e-3, atol=1e-5)
        - 'low': For low precision answers (default: rtol=1e-2, atol=1e-3)
        - 'rough': For approximate answers (default: rtol=1e-1, atol=1e-2)
    
    Returns:
      (candidate, True, precision_level) if a valid candidate is found;
      (candidate, False, None) if a candidate is found but fails validation;
      (None, False, None) if no candidate is found.
    """
    data = params['data']
    
    import re
    import ast
    import numpy as np
    
    # Define default precision levels if not provided
    if precision_levels is None:
        precision_levels = {
            'exact': {'rtol': 1e-10, 'atol': 1e-10},   # Bit-perfect
            'high': {'rtol': 1e-5, 'atol': 1e-8},      # High precision
            'medium': {'rtol': 1e-3, 'atol': 1e-5},    # Medium precision
            'low': {'rtol': 1e-2, 'atol': 1e-3},       # Low precision
            'rough': {'rtol': 1e-1, 'atol': 1e-2}      # Rough approximation
        }
    
    # Helper function to validate matrix dimensions
    def validate_dimensions(result, data_list):
        if not result or not isinstance(result, list) or not all(isinstance(row, list) for row in result):
            return False
        if not data_list or not data_list[0]:
            return False
        
        # For inverse covariance matrix, it should be square with dimensions equal to the number of features (vector length)
        feature_dim = len(data_list[0])
        return len(result) == feature_dim and all(len(row) == feature_dim for row in result)
    
    # Helper function to validate matrix symmetry
    def validate_symmetry(matrix, level='high'):
        matrix_np = np.array(matrix)
        rtol = precision_levels[level]['rtol']
        atol = precision_levels[level]['atol']
        return np.allclose(matrix_np, matrix_np.T, rtol=rtol, atol=atol)
    
    # Helper function to compute expected result for validation
    def compute_expected(data_list):
        try:
            if not data_list or not data_list[0]:
                return None
            
            # Create a numpy array for efficient calculation
            data_array = np.array(data_list)
            # Calculate covariance matrix
            # Subtract mean from each feature
            data_centered = data_array - np.mean(data_array, axis=0)
            # Calculate covariance matrix
            cov_matrix = np.dot(data_centered.T, data_centered) / (data_array.shape[0] - 1)
            
            # Calculate inverse of covariance matrix
            # Adding a small regularization term to handle potential numerical instability
            cov_matrix_reg = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-10
            inv_cov_matrix = np.linalg.inv(cov_matrix_reg)
            
            return inv_cov_matrix.tolist()
        except np.linalg.LinAlgError:
            # Handle case where covariance matrix is singular
            return None
        except:
            return None
    
    # Helper function to check if values match at a given precision level
    def check_precision(candidate_values, expected_values, precision_level):
        candidate_np = np.array(candidate_values)
        expected_np = np.array(expected_values)
        rtol = precision_levels[precision_level]['rtol']
        atol = precision_levels[precision_level]['atol']
        return np.allclose(candidate_np, expected_np, rtol=rtol, atol=atol)
    
    # Helper function to verify matrix inverse property (A * A^-1 ≈ I)
    def verify_inverse_property(candidate_inv, data_list, precision_level='medium'):
        try:
            # Calculate covariance matrix
            data_array = np.array(data_list)
            data_centered = data_array - np.mean(data_array, axis=0)
            cov_matrix = np.dot(data_centered.T, data_centered) / (data_array.shape[0] - 1)
            
            # Convert candidate inverse to numpy array
            inv_cov_np = np.array(candidate_inv)
            
            # Calculate product of covariance matrix and its inverse
            product = np.matmul(cov_matrix, inv_cov_np)
            
            # Check if product is approximately identity matrix
            n = cov_matrix.shape[0]
            identity = np.eye(n)
            
            rtol = precision_levels[precision_level]['rtol']
            atol = precision_levels[precision_level]['atol']
            
            return np.allclose(product, identity, rtol=rtol, atol=atol)
        except:
            return False
    
    # First, try the expected format "Answer:" followed by a nested list
    pattern = r'Answer:\s*(\[\s*\[.+?\]\s*\])'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Take the last match as the final answer
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if validate_dimensions(candidate, data):
                # Verify basic properties of an inverse covariance matrix
                
                # 1. Check if the matrix is symmetric
                if not validate_symmetry(candidate, 'medium'):
                    return candidate, False, None
                
                # 2. Verify matrix inverse property at medium precision
                # This is a stronger validation than just comparing to numpy's inverse
                if not verify_inverse_property(candidate, data, 'medium'):
                    return candidate, False, None
                
                # 3. Compare with expected result at different precision levels
                expected = compute_expected(data)
                if expected is not None:
                    for level in ['exact', 'high', 'medium', 'low', 'rough']:
                        if check_precision(candidate, expected, level):
                            return candidate, True, level
                else:
                    # If we can't compute the expected result but the inverse property holds,
                    # we consider it valid at 'low' precision
                    return candidate, True, 'low'
            return candidate, False, None
        except Exception as e:
            pass
    
    # Fallback: search for any nested list structure in the text
    fallback_pattern = r'\[\s*\[(?:[^\[\]]+)\](?:\s*,\s*\[(?:[^\[\]]+)\])*\s*\]'
    fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
    
    for m in reversed(fallback_matches):
        try:
            candidate = ast.literal_eval(m)
            if validate_dimensions(candidate, data):
                # Verify matrix properties
                if not validate_symmetry(candidate, 'medium'):
                    return candidate, False, None
                
                if not verify_inverse_property(candidate, data, 'medium'):
                    return candidate, False, None
                
                expected = compute_expected(data)
                if expected is not None:
                    for level in ['exact', 'high', 'medium', 'low', 'rough']:
                        if check_precision(candidate, expected, level):
                            return candidate, True, level
                else:
                    return candidate, True, 'low'
            return candidate, False, None
        except Exception:
            continue
    
    return None, False, None

# def extract_fixed_k_subset_sum_from_text(text, original_nums, target, k):
def extract_fixed_k_subset_sum_from_text(text, params, target=None):
    """
    Extracts a candidate subset of exactly k elements from the text (e.g. "[1, 4, 5]").
    Validates:
      - List of integers
      - Subset of original_nums (multiset-aware)
      - Sum equals target
      - List length == k

    Returns:
      (candidate, True)  → valid
      (candidate, False) → invalid format or failed validation
      (None, False)      → no valid candidate found
    """
    original_nums = params['nums']
    if target is None:
        target = params['target']
    k = params['k']
    
    pattern = r'Answer:\s*(\[[^\]]+\])'
    matches = re.findall(pattern, text)
    if matches:
        candidate_str = matches[-1]
        try:
            candidate = ast.literal_eval(candidate_str)
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                if len(candidate) != k:
                    return candidate, False
                original_counter = Counter(original_nums)
                candidate_counter = Counter(candidate)
                valid_subset = all(original_counter[num] >= candidate_counter[num] for num in candidate_counter)
                if valid_subset and sum(candidate) == target:
                    return candidate, True
                else:
                    return candidate, False
        except Exception:
            pass

    # Fallback: search for any bracketed list
    fallback_matches = re.findall(r'\[([^\]]+)\]', text)
    for m in reversed(fallback_matches):
        try:
            candidate_str = "[" + m + "]"
            candidate = ast.literal_eval(candidate_str)
            if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                if len(candidate) != k:
                    return candidate, False
                original_counter = Counter(original_nums)
                candidate_counter = Counter(candidate)
                valid_subset = all(original_counter[num] >= candidate_counter[num] for num in candidate_counter)
                if valid_subset and sum(candidate) == target:
                    return candidate, True
                else:
                    return candidate, False
        except Exception:
            continue
    return None, False

def extract_fixed_k_subset_sum_count_unique_sets_from_text(text, params):
    original_nums = params['nums']
    target = params['target']
    k = params['k']
    unique_count = params['unique_count']
    pattern = re.compile(r'(?m)^Answer:\s*([+-]?\d+)\b')

    matches = pattern.findall(text)          # ['42', '-7', ...]
    ints = list(map(int, matches))           # [42, -7, ...]

    # first one (or None if missing)
    first_int = ints[0] if ints else None
    return unique_count, first_int == unique_count if first_int is not None else False

def extract_inner_product_from_text(text, vector_a, vector_b):
    """
    Extracts the numeric answer after 'Answer:' and validates it is the correct inner product of vector_a and vector_b.
    
    Returns:
      (candidate, True) if a valid candidate is found;
      (candidate, False) if a candidate is found but fails validation;
      (None, False) if no candidate is found.
    """
    candidate = _extract_last_numeric_candidate(text)
    if candidate is None:
        return None, False

    correct_inner_product = sum(a * b for a, b in zip(vector_a, vector_b))
    return candidate, candidate == correct_inner_product

def extract_longest_arithmetic_subsequence_from_text(text, params):
    """
    Extracts a candidate longest arithmetic sequence from text in a robust manner.
    "Answer:" followed by an integer is expected.
    If found, it parses and validates the candidate.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    """
    target = params['target']

    candidate = _extract_last_intlike_candidate(text)
    if candidate is None:
        return None, False
    return candidate, candidate == target

def extract_balloon_bursting_from_text(text, params):
    """
    Extracts a candidate maximum coins value from text for the balloon bursting problem.
    Expected format: "Answer:" followed by an integer.
    If found, it parses and validates the candidate against the correct answer.
    If no valid candidate is found in the expected format, it falls back to scanning the entire text.
    
    Args:
        text: The text output from the model
        params: Dictionary containing the problem parameters including 'target' (correct answer)
    
    Returns:
        (candidate, True) if candidate matches the correct answer;
        (candidate, False) if candidate is found but incorrect;
        (None, False) if no candidate is found.
    """
    target = params['target']
    
    # First, try the expected format "Answer:" followed by a number.
    pattern = r'Answer:\s*(\d+)'
    matches = re.findall(pattern, text)
    if matches:
        # Take the last match as the final answer
        try:
            candidate = int(matches[-1])
            return candidate, candidate == target
        except Exception:
            pass
    
    # Fallback: search the text for any number.
    fallback_matches = re.findall(r'(\d+)', text)
    for m in reversed(fallback_matches):
        try:
            candidate = int(m)
            return candidate, candidate == target
        except Exception:
            continue
    
    return None, False
    if isinstance(original_s, dict):
        original_s = original_s.get("s", original_s.get("original_s", ""))


def extract_max_avg_subarray_variable_k_window_from_text(text, params):
    """
    Extract & evaluate max-average-subarray window from model output.

    Expected params:
      - params["nums"]: list[int]
      - params["k"]: int
      - params["answer"]: list[int]   (the correct window)

    Correct if:
      - candidate is a list of integers
      - length == k
      - candidate is a valid contiguous subarray of nums
      - candidate == params["answer"]  (leftmost optimal window)

    Returns:
      (candidate, True)  if correct
      (candidate, False) if extracted but incorrect
      (None, False)      if no candidate found
    """

    nums = params["nums"]
    k = params["k"]
    gt = params["answer"]

    def is_valid(candidate):
        if not (isinstance(candidate, list) and len(candidate) == k):
            return False

        # check contiguous subarray
        for i in range(len(nums) - k + 1):
            if nums[i:i+k] == candidate:
                return candidate == gt
        return False

    # --- prefer Answer: [...] ---
    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)

    if matches:
        try:
            cand = ast.literal_eval(matches[-1])
            return cand, is_valid(cand)
        except:
            pass

    # --- fallback: any list ---
    fallback = re.findall(r'\[[^\]]*\]', text)
    for s in reversed(fallback):
        try:
            cand = ast.literal_eval(s)
            return cand, is_valid(cand)
        except:
            continue

    return None, False


def extract_max_avg_subarray_fixed_k_window_from_text(text, params):
    """
    Extract & evaluate max-average-subarray window from model output.

    Expected params:
      - params["nums"]: list[int]
      - params["k"]: int
      - params["answer"]: list[int]   (the correct window)

    Correct if:
      - candidate is a list of integers
      - length == k
      - candidate is a valid contiguous subarray of nums
      - candidate == params["answer"]  (leftmost optimal window)

    Returns:
      (candidate, True)  if correct
      (candidate, False) if extracted but incorrect
      (None, False)      if no candidate found
    """

    nums = params["nums"]
    k = params["k"]
    gt = params["answer"]

    def is_valid(candidate):
        if not (isinstance(candidate, list) and len(candidate) == k):
            return False

        # check contiguous subarray
        for i in range(len(nums) - k + 1):
            if nums[i:i+k] == candidate:
                return candidate == gt
        return False

    # --- prefer Answer: [...] ---
    pattern = r'Answer:\s*(\[[^\]]*\])'
    matches = re.findall(pattern, text)

    if matches:
        try:
            cand = ast.literal_eval(matches[-1])
            return cand, is_valid(cand)
        except:
            pass

    # --- fallback: any list ---
    fallback = re.findall(r'\[[^\]]*\]', text)
    for s in reversed(fallback):
        try:
            cand = ast.literal_eval(s)
            return cand, is_valid(cand)
        except:
            continue

    return None, False