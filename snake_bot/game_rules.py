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
    return any(max(abs(position[0] - bomb[0]), abs(position[1] - bomb[1])) <= radius
               for bombs, radius in ((bombs_large, 2), (bombs_small, 1))
               for bomb in bombs)


def blast_zone_cells(rows, cols, bombs_large=(), bombs_small=()):
    """Materialize the same bounded Chebyshev zones once for path searches."""
    return frozenset(
        (row, col)
        for bombs, kind in ((bombs_large, 'large'), (bombs_small, 'small'))
        for bomb in bombs
        for row in range(max(0, bomb[0] - BOMB_RADII[kind]),
                         min(rows, bomb[0] + BOMB_RADII[kind] + 1))
        for col in range(max(0, bomb[1] - BOMB_RADII[kind]),
                         min(cols, bomb[1] + BOMB_RADII[kind] + 1)))


def score_after_bomb(score):
    """Bombs replace the ordinary move reward, independently of crash history."""
    return 0 if score > 0 else score - CRASH_PENALTY


def bombs_active_on_turn(remaining_moves, actor_is_next, move_number=1):
    """Project the global clock for one actor's nth alternating move."""
    elapsed = 2 * (move_number - 1) + (0 if actor_is_next else 1)
    return remaining_moves - elapsed > BOMB_EXPIRY_MOVES


def shot_food_reward(multiplier=1):
    return SHOT_FOOD_REWARD * max(1, multiplier)


def bomb_respawn_candidates(rows, cols, bomb_type, heads, occupied=(), edible=()):
    """Eligible observable placements; the caller owns any random selection."""
    radius = BOMB_RADII[bomb_type]
    blocked = set(occupied)
    protected = tuple(heads) + tuple(edible)
    return tuple((row, col) for row in range(rows) for col in range(cols)
                 if (row, col) not in blocked and all(
                     max(abs(row - cell[0]), abs(col - cell[1])) > radius
                     for cell in protected))


def first_shot_hit(head, delta, rows, cols, walls, rival):
    """Only walls, board edges and the first rival cell stop the ray."""
    rival_cells = set(rival)
    row, col = head
    dr, dc = delta
    row, col = row + dr, col + dc
    while 0 <= row < rows and 0 <= col < cols:
        cell = (row, col)
        if cell in walls:
            return None
        if cell in rival_cells:
            return cell
        row, col = row + dr, col + dc
    return None


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


def shoot_directions(head, rows, cols, walls, rival):
    """Useful rays only: misses and hits on a lone head cannot detach cells."""
    if head is None or len(rival) <= 1:
        return ()
    directions = (('up', (-1, 0)), ('down', (1, 0)),
                  ('left', (0, -1)), ('right', (0, 1)))
    return tuple(direction for direction, delta in directions
                 if first_shot_hit(head, delta, rows, cols, walls, rival) is not None)


def star_spawn_candidates(rows, cols, heads, occupied=(), edible=(),
                          bombs_large=(), bombs_small=(), frenzy=False):
    """All equally eligible cells; selecting server RNG outcomes is caller-owned."""
    blocked = set(occupied) | set(edible) | set(bombs_large) | set(bombs_small)
    cells = tuple((r, c) for r in range(rows) for c in range(cols)
                  if (r, c) not in blocked and not within_blast_zone(
                      (r, c), bombs_large, bombs_small))
    if frenzy:
        return cells
    if len(heads) != 2 or any(head is None for head in heads):
        return ()
    def distance(cell, head):
        return abs(cell[0] - head[0]) + abs(cell[1] - head[1])
    tied = [(distance(cell, heads[0]), cell) for cell in cells
            if distance(cell, heads[0]) == distance(cell, heads[1])]
    nearest = min((value for value, _ in tied), default=None)
    return tuple(cell for value, cell in tied if value == nearest)
