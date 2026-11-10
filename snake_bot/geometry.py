"""Shared coordinate operations for historical boards and the V11 cube."""

from itertools import product
from functools import lru_cache

PLANAR_DIRECTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
CUBE_DIRECTIONS = {
    'up': (0, -1, 0),
    'down': (0, 1, 0),
    'left': (0, 0, -1),
    'right': (0, 0, 1),
    'ascend': (-1, 0, 0),
    'descend': (1, 0, 0),
}


def directions_for(value):
    """Choose directions from a state or the arity of one coordinate."""
    if hasattr(value, 'levels'):
        return CUBE_DIRECTIONS if value.levels > 1 else PLANAR_DIRECTIONS
    return CUBE_DIRECTIONS if len(value) == 3 else PLANAR_DIRECTIONS


def dimension_sizes(state):
    """Return axis lengths in the same order as coordinates."""
    planar = (state.rows, state.cols)
    return (state.levels,) + planar if state.levels > 1 else planar


def iter_cells(rows, cols, levels=1):
    """Enumerate true board cells, keeping legacy coordinates two-dimensional."""
    sizes = (levels, rows, cols) if levels > 1 else (rows, cols)
    return product(*(range(size) for size in sizes))


def translated(position, delta):
    """Translate a coordinate without flattening or discarding an axis."""
    return tuple(value + offset for value, offset in zip(position, delta))


def step(position, direction):
    """Return the neighboring coordinate, including out-of-bounds targets."""
    return translated(position, directions_for(position)[direction])


def bounded(position, sizes):
    """Check both coordinate arity and every dimension's bounds."""
    return len(position) == len(sizes) and all(
        0 <= value < size for value, size in zip(position, sizes)
    )


def neighbors(position, rows, cols, levels=None):
    """Yield at most four flat or six cube neighbors within the board."""
    sizes = neighbor_sizes(position, rows, cols, levels)
    return iter(_bounded_neighbors(tuple(position), sizes))


@lru_cache(maxsize=8192)
def _bounded_neighbors(position, sizes):
    """Cache immutable geometry while leaving observation and occupancy filters fresh."""
    return tuple(cell for cell in adjacent_coordinates(position) if bounded(cell, sizes))


def adjacent_coordinates(position):
    """Yield immediate targets without bounds filtering."""
    return (translated(position, delta) for delta in directions_for(position).values())


def neighbor_sizes(position, rows, cols, levels):
    """Resolve legacy or explicitly supplied cube dimensions."""
    if len(position) == 2:
        return rows, cols
    return levels or 20, rows, cols


def manhattan(first, second):
    """Return orthogonal distance over every coordinate axis."""
    return sum(abs(a - b) for a, b in zip(first, second))


def chebyshev(first, second):
    """Return the cube/square blast distance including diagonal movement."""
    return max(abs(a - b) for a, b in zip(first, second))


def plane_contains(head, cell):
    """Whether a cube cell lies on at least one plane through a head."""
    return any(a == b for a, b in zip(head, cell))
