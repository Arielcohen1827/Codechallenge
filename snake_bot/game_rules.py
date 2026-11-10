"""Authoritative scoring and placement geometry shared by runtime and simulation."""

from itertools import product

from snake_bot.geometry import bounded, chebyshev, directions_for, iter_cells, manhattan
from snake_bot.geometry import plane_contains, translated

FOOD_SCORE = 100
WRONG_FOOD_PENALTY = -500
MULTIPLIER_PICKUP_SCORE = 50
MAX_MOVES = 400
CRASH_KEEP_LENGTH = 3
CRASH_PENALTY = 500
ELIMINATION_SCORE_THRESHOLD = -2500
TAIL_FOOD_BASE_REWARD = 100
RIVAL_CRASH_REWARD = 0
BOMB_EXPIRY_MOVES = 100
BOMB_RADII = {'large': 2, 'small': 1}
SHOOT_COST = 5000
INVALID_ACTION_PENALTY = -500
SHOT_FOOD_REWARD = 100
STAR_TURNS = 50
STAR_RESPAWN_ROUNDS = 20
STAR_FRENZY_COUNT = 10
MAX_TIMEOUTS = 10


def within_blast_zone(position, bombs_large=(), bombs_small=()):
    """Chebyshev distance includes diagonal cells and the bomb itself."""
    return any(
        chebyshev(position, bomb) <= radius
        for bombs, radius in ((bombs_large, 2), (bombs_small, 1))
        for bomb in bombs
    )


def blast_zone_cells(rows, cols, bombs_large=(), bombs_small=(), levels=1):
    """Materialize the same bounded Chebyshev zones once for path searches."""
    zones = (
        bomb_zone(bomb, radius, rows, cols, levels)
        for bombs, radius in ((bombs_large, 2), (bombs_small, 1))
        for bomb in bombs
    )
    return frozenset().union(*zones)


def bomb_zone(bomb, radius, rows, cols, levels=1):
    """Enumerate only the local square/cube around one bomb."""
    sizes = (rows, cols) if len(bomb) == 2 else (levels if levels > 1 else 20, rows, cols)
    ranges = (
        range(max(0, value - radius), min(size, value + radius + 1))
        for value, size in zip(bomb, sizes)
    )
    return tuple(product(*ranges))


def score_after_bomb(score):
    """Bombs replace the ordinary move reward, independently of crash history."""
    return 0 if score > 0 else score - CRASH_PENALTY


def bombs_active_on_turn(remaining_moves, actor_is_next, move_number=1):
    """Project the global clock for one actor's nth alternating move."""
    elapsed = 2 * (move_number - 1) + (0 if actor_is_next else 1)
    return remaining_moves - elapsed > BOMB_EXPIRY_MOVES


def shot_food_reward(multiplier=1):
    """Compute freely ordered shot-food points with the current multiplier."""
    return SHOT_FOOD_REWARD * max(1, multiplier)


def _free_cells(rows, cols, blocked, levels=1):
    """Enumerate true board coordinates outside the supplied occupied set."""
    return (cell for cell in iter_cells(rows, cols, levels) if cell not in blocked)


def _outside_blast_radius(position, protected, radius):
    """Require every protected cell to lie outside the bombs full-dimensional reach."""
    return all(chebyshev(position, cell) > radius for cell in protected)


def bomb_respawn_candidates(rows, cols, bomb_type, heads, occupied=(), edible=(), levels=1):
    """Eligible observable placements; the caller owns any random selection."""
    radius = BOMB_RADII[bomb_type]
    protected = tuple(heads) + tuple(edible)
    return tuple(
        cell
        for cell in _free_cells(rows, cols, set(occupied), levels)
        if _outside_blast_radius(cell, protected, radius)
    )


def first_shot_hit(head, delta, rows, cols, walls, rival, levels=20):
    """Only walls, board edges and the first rival cell stop the ray."""
    rival_cells = set(rival)
    sizes = (rows, cols) if len(head) == 2 else (levels, rows, cols)
    cell = translated(head, delta)
    while bounded(cell, sizes):
        if cell in walls:
            return None
        if cell in rival_cells:
            return cell
        cell = translated(cell, delta)
    return None


def cyclic_digit(value, offset=1):
    """Advance within digits one through nine, wrapping across the sequence boundary."""
    return ((int(value) - 1 + offset) % 9) + 1


def find_next_food_digit(values):
    """Infer the ordered target from a complete active digit window."""
    present = {int(value) for value in values}
    if not present:
        return None
    return min(_sequence_starts(present) or present)


def _sequence_starts(digits):
    """Find present digits whose cyclic predecessor is absent."""
    return [digit for digit in digits if cyclic_digit(digit, -1) not in digits]


def food_points(digit=None, multiplier=1):
    """Compute ordinary or digit food points with a minimum multiplier of one."""
    base = FOOD_SCORE if digit is None else digit * FOOD_SCORE
    return base * max(1, multiplier)


def score_after_crash(score, crashed=None):
    """Unknown history is usable only when both histories give the same score."""
    if type(crashed) not in (type(None), bool):
        raise ValueError('Crash history must be True, False or None')
    if score <= 0:
        return score - CRASH_PENALTY
    return _positive_score_after_crash(score, crashed)


def _positive_score_after_crash(score, crashed):
    """Require known crash history before scoring a positive-score crash."""
    if crashed is None:
        raise ValueError('Positive-score crash requires known crash history')
    return score - CRASH_PENALTY if crashed else 0


def tail_food_reward(multiplier=1):
    """Compute beneficiary-only tail food points from the current multiplier."""
    return TAIL_FOOD_BASE_REWARD * max(1, multiplier)


def crash_scores(crashed_score, rival_score, crashed=None):
    """Compatibility wrapper; callers must supply history for positive scores."""
    return score_after_crash(crashed_score, crashed), rival_score


def shoot_directions(head, rows, cols, walls, rival):
    """Useful rays only: misses and hits on a lone head cannot detach cells."""
    if head is None or len(rival) <= 1:
        return ()
    directions = directions_for(head).items()
    return tuple(
        direction
        for direction, delta in directions
        if first_shot_hit(head, delta, rows, cols, walls, rival) is not None
    )


def star_spawn_candidates(
    rows,
    cols,
    heads,
    occupied=(),
    edible=(),
    bombs_large=(),
    bombs_small=(),
    frenzy=False,
    levels=1,
):
    """All equally eligible cells; selecting server RNG outcomes is caller-owned."""
    blocked = set(occupied) | set(edible) | set(bombs_large) | set(bombs_small)
    cells = tuple(
        cell
        for cell in _free_cells(rows, cols, blocked, levels)
        if not within_blast_zone(cell, bombs_large, bombs_small)
    )
    if frenzy:
        return cells
    return _equidistant_star_cells(cells, heads)


def _equidistant_star_cells(cells, heads):
    """Select nearest equally distant eligible cells for two known heads."""
    if len(heads) != 2 or None in heads:
        return ()
    tied = _tied_star_distances(cells, *heads)
    nearest, _ = min(tied, default=(None, None))
    return tuple(cell for value, cell in tied if value == nearest)


def _tied_star_distances(cells, first, second):
    """Measure candidate star races using full coordinates and mutual visibility."""
    return [
        (_manhattan(cell, first), cell)
        for cell in cells
        if _manhattan(cell, first) == _manhattan(cell, second)
        and star_visible_to_both(cell, first, second)
    ]


def _manhattan(cell, head):
    """Measure orthogonal distance consistently with the shared geometry."""
    return manhattan(cell, head)


def star_visible_to_both(cell, first, second):
    """Normal cube stars must intersect at least one plane of each head."""
    if len(cell) == 2:
        return True
    return plane_contains(first, cell) and plane_contains(second, cell)
