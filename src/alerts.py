"""Cost-based maintenance alerting.

Decision at each monitoring point: raise an alert if the predicted RUL
(lower bound or median) is <= threshold T.
  * Missed failure: no alert but true RUL <= LEAD (not enough time to schedule)  -> C_FAIL
  * False alarm   : alert but true RUL  > LEAD (maintenance done too early)       -> C_EARLY
T is chosen on VALIDATION engines only, then applied unchanged to the test set.
"""
import numpy as np

LEAD, C_FAIL, C_EARLY = 30, 10.0, 1.0


def total_cost(pred, true, T, lead=LEAD, c_fail=C_FAIL, c_early=C_EARLY):
    alert = pred <= T
    danger = true <= lead
    miss = (~alert) & danger
    false_alarm = alert & (~danger)
    return c_fail * miss.sum() + c_early * false_alarm.sum(), int(miss.sum()), int(false_alarm.sum())


def choose_threshold(pred, true, grid=None, **kw):
    grid = np.arange(0, 126, 1) if grid is None else grid
    costs = [total_cost(pred, true, T, **kw)[0] for T in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate_policy(pred, true, T, **kw):
    cost, miss, fa = total_cost(pred, true, T, **kw)
    n = len(true)
    return {"threshold": T, "cost_per_engine": cost / n, "missed_failures": miss,
            "false_alarms": fa, "n_engines": n}
