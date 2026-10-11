FOOD_SCORE = 100
WRONG_FOOD_PENALTY = -500
MULTIPLIER_PICKUP_SCORE = 50
MAX_MOVES = 400
CRASH_KEEP_LENGTH = 3
CRASH_PENALTY = 500
ELIMINATION_SCORE_THRESHOLD = -2500
TAIL_FOOD_BASE_REWARD = 100
RIVAL_CRASH_REWARD = 0


def cyclic_digit(value, offset=1):
    return ((int(value) - 1 + offset) % 9) + 1


def find_next_food_digit(values):
    present = {int(value) for value in values}
    if not present:
        return None
    candidates = [digit for digit in present if cyclic_digit(digit, -1) not in present]
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates or present)


def food_points(digit=None, multiplier=1):
    base = FOOD_SCORE if digit is None else digit * FOOD_SCORE
    return base * max(1, multiplier)


def score_after_crash(score, crashed=None):
    """Unknown history is usable only when both histories give the same score."""
    if crashed is not None and type(crashed) is not bool:
        raise ValueError('Crash history must be True, False or None')
    if score > 0:
        if crashed is None:
            raise ValueError('Positive-score crash requires known crash history')
        if not crashed:
            return 0
    return score - CRASH_PENALTY


def tail_food_reward(multiplier=1):
    return TAIL_FOOD_BASE_REWARD * max(1, multiplier)


def crash_scores(crashed_score, rival_score, crashed=None):
    """Compatibility wrapper; callers must supply history for positive scores."""
    return score_after_crash(crashed_score, crashed), rival_score
