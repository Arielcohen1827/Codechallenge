import ast
from contextlib import contextmanager
from dataclasses import asdict, replace
import importlib
from pathlib import Path
import sys
from types import ModuleType
import typing
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent


def reject_union(left, right):
    raise TypeError('Python 3.9 cannot evaluate type | None')


class AnnotationUnionShim(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.BitOr):
            return ast.copy_location(ast.Call(
                func=ast.Name(id='__python39_union__', ctx=ast.Load()),
                args=[node.left, node.right], keywords=[]), node)
        return node


class Python39Annotations(ast.NodeTransformer):
    def visit_AnnAssign(self, node):
        # Change only annotations; preserve runtime set unions and game logic.
        node.annotation = AnnotationUnionShim().visit(node.annotation)
        return self.generic_visit(node)


@contextmanager
def legacy_module(name, source=None):
    filename = (ROOT / (name.replace('.', '/') + '.py')
                if source is None else Path(__file__))
    tree = ast.parse(source if source is not None else filename.read_text(encoding='utf-8'),
                     filename=str(filename), feature_version=9)
    tree = ast.fix_missing_locations(Python39Annotations().visit(tree))
    isolated_name = '_python39_' + name.replace('.', '_')
    module = ModuleType(isolated_name)
    module.__file__ = str(filename)
    module.__package__ = name.rpartition('.')[0]
    module.__dict__['__python39_union__'] = reject_union
    with patch.dict(sys.modules, {isolated_name: module}):
        exec(compile(tree, str(filename), 'exec', dont_inherit=True), module.__dict__)
        yield module


class Python39CompatibilityTests(unittest.TestCase):
    def test_shim_rejects_eager_unions_but_preserves_postponed_dataclass_behavior(self):
        source = ('from dataclasses import dataclass\n'
                  '@dataclass\nclass Result:\n    value: int | None = None\n')
        with self.assertRaisesRegex(TypeError, 'Python 3.9'):
            with legacy_module('fixture', source):
                pass
        with legacy_module('fixture', 'from __future__ import annotations\n' + source) as module:
            for value in (None, 7):
                result = module.Result(value)
                self.assertEqual(asdict(result), {'value': value})
                self.assertEqual(replace(result, value=9).value, 9)

    def test_state_import_with_legacy_annotations_preserves_food_and_normal_moves(
        self,
    ):
        with legacy_module('snake_bot.snake_state') as module:
            for board, score, digit in [
                ('|A* B|\n|    |', 100, None),
                ('|A  B|\n|    |', 1, None),
                ('|A12B|\n|    |', 100, 2),
            ]:
                with self.subTest(board=board):
                    state = module.parse_state(
                        {'board': board, 'side': 'A', 'remaining_moves': 10}
                    )
                    moved = module.apply_move(state, 'right', 'A')
                    self.assertEqual(moved.head('A'), (0, 1))
                    self.assertEqual(moved.scores['A'], score)
                    self.assertEqual(moved.next_food_digit, digit)
                    self.assertEqual(
                        asdict(replace(moved, remaining_moves=3))[
                            'remaining_moves'
                        ],
                        3,
                    )

    def test_planner_import_with_legacy_annotations_preserves_path_direction(self):
        with legacy_module('snake_bot.food_planner') as module:
            for path, enemy_distance, expected in [(((0, 0), (0, 1)), 2, 'right'),
                                                   (((0, 0),), None, None)]:
                plan = module.FoodPlan(food=path[-1], path=path, our_distance=len(path) - 1,
                                       enemy_distance=enemy_distance, race_margin=0,
                                       space_after=8, exits_after=2, tail_reachable_after=True,
                                       next_food_distance=None, safety='SAFE', value=100)
                self.assertEqual(plan.first_move, expected)
                self.assertEqual(asdict(plan)['enemy_distance'], enemy_distance)
                self.assertEqual(replace(plan, value=200).first_move, expected)

    def test_lab_import_with_legacy_annotations_preserves_simulation_and_summary(self):
        with legacy_module('tools.bot_lab') as module:
            for candidate_side in (None, 'A'):
                result = module.simulate_game(7, max_turns=0, rows=5, cols=6,
                                              candidate_side=candidate_side)
                self.assertEqual(asdict(result)['candidate_side'], candidate_side)
                self.assertEqual(result.turns, 0)
                self.assertIsNone(result.winner)
                summary = module.summarize_simulations([result])
                self.assertEqual(summary['games'], 1)
                self.assertEqual(summary['draws'], 1)
                self.assertEqual(summary['candidate_games'], int(candidate_side == 'A'))

    @unittest.skipIf(
        sys.version_info < (3, 10),
        'PEP 604 runtime type-hint resolution requires Python 3.10',
    )
    def test_native_type_hint_resolution_still_supports_dataclass_consumers(
        self,
    ):
        cases = [
            (
                'snake_bot.snake_state',
                'GameState',
                {'next_food_digit': typing.Optional[int]},
            ),
            (
                'snake_bot.food_planner',
                'FoodPlan',
                {
                    'enemy_distance': typing.Optional[int],
                    'next_food_distance': typing.Optional[int],
                },
            ),
            (
                'tools.bot_lab',
                'SimResult',
                {
                    'candidate_side': typing.Optional[str],
                    'winner': typing.Optional[str],
                    'no_legal_side': typing.Optional[str],
                },
            ),
        ]
        for module_name, class_name, expected in cases:
            cls = getattr(importlib.import_module(module_name), class_name)
            resolved = typing.get_type_hints(cls)
            for field, hint in expected.items():
                self.assertEqual(resolved[field], hint)
