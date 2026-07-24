"""Legacy compatibility helpers for utils.py.

The flattened-dataset evaluation path in ``new_complexity`` does not rely on the
older generator stack from the original repository. Keeping only the small
runtime helpers here avoids copying the original, much larger support files.
"""

from __future__ import annotations


def check_regrex(s: str, p: str) -> bool:
    n, m = len(s), len(p)
    dp = [[-1 for _ in range(m + 1)] for _ in range(n + 1)]

    def dp_f(i: int, j: int) -> int:
        if dp[i][j] != -1:
            return dp[i][j]
        if i >= n and j >= m:
            return 1
        if j >= m:
            return 0

        match = 0
        if i < n and (s[i] == p[j] or p[j] == "."):
            match = 1

        if j < m - 1 and p[j + 1] == "*":
            if match:
                dp[i][j] = 1 if dp_f(i + 1, j) or dp_f(i, j + 2) else 0
            else:
                dp[i][j] = 1 if dp_f(i, j + 2) else 0
            return dp[i][j]

        if match:
            dp[i][j] = dp_f(i + 1, j + 1)
            return dp[i][j]

        dp[i][j] = 0
        return 0

    return dp_f(0, 0) == 1


def check_sat(cnf_clauses):
    max_var = 0
    for clause in cnf_clauses:
        for lit in clause:
            max_var = max(max_var, abs(lit))

    assignment = [None] * (max_var + 1)

    def evaluate_clause(clause, current_assignment):
        for lit in clause:
            var = abs(lit)
            if current_assignment[var] is None:
                return None
            if (lit > 0 and current_assignment[var]) or (lit < 0 and not current_assignment[var]):
                return True
        return False

    def evaluate_formula(current_assignment):
        for clause in cnf_clauses:
            result = evaluate_clause(clause, current_assignment)
            if result is False:
                return False
            if result is None:
                return None
        return True

    def backtrack(var_idx: int) -> bool:
        if var_idx > max_var:
            return True

        assignment[var_idx] = True
        formula_eval = evaluate_formula(assignment)
        if formula_eval is True:
            return True
        if formula_eval is None and backtrack(var_idx + 1):
            return True

        assignment[var_idx] = False
        formula_eval = evaluate_formula(assignment)
        if formula_eval is True:
            return True
        if formula_eval is None and backtrack(var_idx + 1):
            return True

        assignment[var_idx] = None
        return False

    return backtrack(1)


__all__ = ["check_regrex", "check_sat"]
