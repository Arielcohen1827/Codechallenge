"""Decode the three authoritative V11 planes and preserve observation uncertainty."""

from dataclasses import dataclass, field

from snake_bot.geometry import neighbors


def plane_rows(text):
    """Read Unicode code points while removing only the visual row borders."""
    return tuple(line[1:-1] for line in text.splitlines() if line.startswith('|'))


def observation_planes(data):
    """Prefer explicit planes, falling back to combined x/y/z protocol order."""
    combined = str(data.get('board', '')).replace('\r', '').split('\n\n')
    names = ('board_x', 'board_y', 'board_z')
    return {
        name: plane_rows(data.get(name, combined[index] if index < len(combined) else ''))
        for index, name in enumerate(names)
    }


def plane_coordinate(name, first, second, head):
    """Map one visual cell into level/row/column coordinates."""
    return {
        'board_x': (first, second, head[2]),
        'board_y': (first, head[1], second),
        'board_z': (head[0], first, second),
    }[name]


def add_plane(cells, name, lines, head):
    """Merge one plane, rejecting contradictory intersections."""
    for first, line in enumerate(lines):
        for second, symbol in enumerate(line):
            cell = plane_coordinate(name, first, second, head)
            add_observed_cell(cells, cell, symbol)


def add_observed_cell(cells, cell, symbol):
    """Keep duplicate coordinates only when their observed symbols agree."""
    if cell in cells and cells[cell] != symbol:
        raise ValueError('Conflicting symbols at a cube-plane intersection')
    cells[cell] = symbol


@dataclass(frozen=True)
class CubeObservation:
    """One server observation, including visible empty cells and original planes."""

    head: tuple
    cells: dict
    planes: dict
    levels: int = 20
    rows: int = 20
    cols: int = 20

    @property
    def visible_cells(self):
        """Return only coordinates actually supplied in the three planes."""
        return frozenset(self.cells)

    @classmethod
    def from_turn_data(cls, data):
        """Decode a V11 payload without constructing hidden cube contents."""
        raw_head = data['head']
        head = tuple(int(raw_head[name]) for name in ('level', 'row', 'col'))
        planes = observation_planes(data)
        cells = {}
        for name, lines in planes.items():
            add_plane(cells, name, lines, head)
        return cls(
            head,
            cells,
            planes,
            int(data.get('levels', 20)),
            int(data.get('rows', 20)),
            int(data.get('cols', 20)),
        )

    def positions(self, symbols):
        """Return observed cells carrying one of the requested symbols."""
        return frozenset(cell for cell, symbol in self.cells.items() if symbol in symbols)


@dataclass(frozen=True)
class RememberedCell:
    """An off-plane object is a historical observation, never confirmed occupancy."""

    symbol: str
    last_seen_turn: int


@dataclass(frozen=True)
class KnowledgeState:
    """Keep current observations separate from timestamped off-plane memories."""

    observed: dict = field(default_factory=dict)
    remembered: dict = field(default_factory=dict)
    last_enemy_head: tuple | None = None
    last_enemy_turn: int | None = None
    next_food_digit: int | None = None
    observed_turn: int = 0

    def reconcile(self, observation, turn):
        """Invalidate reobserved cells and retain only uncertain off-plane objects."""
        memories = dict(self.remembered)
        memories.update(
            {
                cell: RememberedCell(symbol, self.observed_turn)
                for cell, symbol in self.observed.items()
                if symbol != ' '
            }
        )
        for cell in observation.cells:
            memories.pop(cell, None)
        return KnowledgeState(
            dict(observation.cells),
            memories,
            self.last_enemy_head,
            self.last_enemy_turn,
            self.next_food_digit,
            turn,
        )


def observed_body(observation, side, invincible, previous=(), explicit_head=None):
    """Order only a connected observed body prefix; retain other segments separately."""
    cells = set(observation.positions((side, side.lower())))
    head_symbol = side.lower() if invincible else side
    heads = observation.positions(head_symbol)
    head = explicit_head if explicit_head is not None else next(iter(heads), None)
    if head is None:
        return ()
    return connected_prefix(head, cells, previous, observation)


def connected_prefix(head, cells, previous, observation):
    """Follow confirmed adjacent segments, stopping when branching is ambiguous."""
    ordered = [head]
    remaining = cells - {head}
    while remaining:
        adjacent = (
            set(neighbors(ordered[-1], observation.rows, observation.cols, observation.levels))
            & remaining
        )
        following = prefix_successor(ordered, adjacent, previous)
        if following is None:
            break
        ordered.append(following)
        remaining.remove(following)
    return tuple(ordered)


def prefix_successor(ordered, adjacent, previous):
    """Use a unique visible neighbor or a still-confirmed historical ordering."""
    if len(adjacent) == 1:
        return next(iter(adjacent))
    if ordered[-1] in previous:
        index = previous.index(ordered[-1]) + 1
        if index < len(previous) and previous[index] in adjacent:
            return previous[index]
    return None
