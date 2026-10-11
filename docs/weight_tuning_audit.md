# Weight Tuning Audit for v7

Audit date: 2026-10-11. HEAD: `4650f4739f9bc8cae339af10c8659a1b5ec5cf01`.

Documentation only: no production, rules, strategy, active weights, optimizer, logs or credentials changed. No tests, benchmarks or optimizer runs. Governance MEDIUM; explicitly authorized audit/report. Strict TDD does not apply to this documentation change. Both project skill registries are absent; no domain skill application or OPSX action was needed. The working tree already contained untracked `tests/test_benchmark_bot_v7.py`; it was preserved. `.coveragerc` has `source = .` and `fail_under = 90`, with tests/venv/cache/git excluded; no coverage run performed.

## Counts and Effective Values

147 declared weights: 135 decision-active and 12 unused by the current decision flow. 65 existing optimizer ranges, all decision-active. Active JSON has 37 explicit keys: 33 differ from defaults, four equal defaults; 110 keys inherit defaults. Proposed initial perturbation subset: 16; freeze 131 at effective current values.

Effective means `DEFAULT_WEIGHTS` merged with known integer keys in `weights/active_weights.json`, matching `load_weights` and `set_active_weights` (bot_weights.py:167-175). A bare import starts with defaults (line 155); production startup loads active weights at run.py:423, and optimizer baseline loads its --base file at tools/optimize_weights.py:296. This audit does not introspect another running process.

## Proposed Initial Bounds

These are conservative search hypotheses, not demonstrated performance improvements. Each interval is symmetric around the actual effective current integer, with radius floor(10% of its absolute value), minimum one. Replace old optimizer bounds for this subset rather than intersecting: `food_distance_cost = 899` sits at the old maximum 900, so preserving the old maximum would make an almost one-sided search. Recommended [810,988] deliberately exceeds that historical limit. The race_margin_value interval [299,365] also extends one integer below its historical minimum 300. These proposals intentionally use current-centered bounds; they are not intersections with historical ranges. All other proposed intervals stay inside historical ranges.

Use one changed weight per candidate initially, same seeds/scenarios/budgets on both sides, and preserve the complete effective baseline. Keep active weights unchanged; no automatic activation. Wider or joint perturbations require comparative evidence. Initial subset avoids safety penalties, branch thresholds, economic projections, advantage activation, endgame controls and search budgets.

```json
{
  "schema_version": 1,
  "root": "C:/Users/ariel/OneDrive/Desktop/Computacion Bot/Codechallenge",
  "head": "4650f4739f9bc8cae339af10c8659a1b5ec5cf01",
  "counts": {
    "total": 147,
    "decision_active": 135,
    "unused": 12,
    "unused_dormant": 11,
    "unused_no_reference": 1,
    "existing_tunable": 65,
    "explicit_active": 37,
    "inherited_default": 110,
    "changed_from_default": 33,
    "recommended_perturbable": 16,
    "recommended_frozen": 131
  },
  "registry": "absent: .agents/SKILLS.md and .atl/skill-registry.md",
  "status_definition": "decision-active means reachable from current SnakeBrain decision flow; unused includes dormant helpers, even with literal references",
  "bounds_policy": "integer current +/- max(1,floor(abs(current)*0.10)); replace historical optimizer bounds, do not intersect; freeze all other weights",
  "validated_performance": false,
  "recommended_ranges": {
    "food_distance_cost": [
      810,
      988
    ],
    "race_margin_value": [
      299,
      365
    ],
    "hunger_value": [
      210,
      256
    ],
    "space_after_value": [
      11,
      13
    ],
    "exits_after_value": [
      414,
      504
    ],
    "tail_reachable_bonus": [
      171,
      209
    ],
    "center_unit_value": [
      48,
      58
    ],
    "edge_depth_value": [
      417,
      509
    ],
    "next_food_value": [
      294,
      358
    ],
    "chain_food_value": [
      234,
      286
    ],
    "chain_race_value": [
      162,
      198
    ],
    "chain_close_food_bonus": [
      810,
      990
    ],
    "hunger_close_food_value": [
      81,
      99
    ],
    "current_target_bonus": [
      749,
      915
    ],
    "fallback_denial_progress": [
      108,
      132
    ],
    "fallback_cycle_penalty": [
      339,
      413
    ]
  }
}
```

## Inactive Call Chain

`adjacent_plan_tactical_override_margin` has no core read. Eleven other weights are read only by `survival_position_score` (food_planner.py:1530), `survival_move_analysis` (1573) and `rank_survival_moves` (1675). Those helpers call each other, but current SnakeBrain imports rank_survival_moves without calling it; fallback uses `rank_deep_moves` (snake_brain.py:388). Searches of snake_bot/, tools/ and run.py found no external production call into this helper chain. Freeze these 12 for current optimizer work. The shared territory/control/fallback penalties also used by live functions remain decision-active. Inactivity is static reachability for the inspected current flow, not a claim about arbitrary external callers or tests.

## Structural Coupling and Redundancy

No statistical correlation has been measured. These groups share inputs or interact algebraically; none is proven interchangeable:

| Group | Weights | Structural evidence / implication |
|---|---|---|
| Distance / tempo / race | food_distance_cost, race_margin_value, hunger_value, numbered_food_hunger_value, hunger_close_food_value, chain_food_value, chain_race_value, next_food_value; fallback_progress_value, fallback_hunger_progress_value, fallback_food_distance_cost | Planner combines distance, race and hunger (food_planner.py:929-941); fallback and deep scoring reuse progress-distance signals (food_planner.py:1716-1725; move_search.py:282-286). Tune separately first; jointly changing them confounds attribution. |
| Center / edge | center_unit_value, edge_depth_value, center_edge0_penalty, center_edge1_penalty, position_edge0_penalty, position_edge1_penalty, position_edge2_penalty, position_corner_penalty, voronoi_center_value | Center score adds center distance and capped edge depth, with edge penalties (food_planner.py:599-605); position penalties reuse geometry (845-860). position_edge1/2 also set arbitration gates (snake_brain.py:319,332,364). |
| Space / mobility / tail | space_after_value, exits_after_value, tail_reachable_bonus, deep_region_value, deep_exit_value, deep_reply_value, deep_tail_bonus, deep_small_region_penalty, small_region_penalty, one_exit_penalty, territory_control_value | Same or related region/mobility features occur at several decision layers (food_planner.py:932-934,1425-1452,1721; move_search.py:220-241). Saturation and body-length gates prevent exact equivalence. Dormant survival_* analogues do not count as active dimensions. |
| Economy / sequence | objective_point_value, economic_projection_value, economic_projection_cap, multiplier_future_point_value, multiplier_denial_point_value, multiplier_gap_value, sequence_reward_point_value, sequence_setup_reward_point_value | Immediate reward, future income and economic swing contribute to the same plan (food_planner.py:906-928,1044,1053-1055). Cap/value coupling can hide perturbations through saturation; multiplier and sequence activation depend on state. |
| Denial / advantage | rival_food_threat_value, rival_chain_threat_value, rival_reward_point_value, denial_food_bonus, denial_chain_bonus, rival_chain_ignore_penalty; advantage_threat_ratio, advantage_score_buffer, advantage_min_lead, advantage_max_activation_lead | Denial weights participate in scoring and thresholds/caps (food_planner.py:355-375,1153); advantage lead threshold clamps dynamic threat (406-420). Maintain activation relationships and freeze this group initially. |
| Safety / overrides / search mixing | zero_escape_penalty, delayed_zero_penalty, one_escape_penalty, dangerous_penalty, fallback_danger_penalty, deep_* safety penalties, deep_hot_lost_multiplier, hot_lost_food_base, hot_lost_food_enemy_bonus; current_target_bonus, food_plan_switch_margin, deep_plan_override_margin; iterative_search_weight_percent, iterative_search_adjustment_cap | Penalties affect summed rankings and fixed pressure thresholds; deep hot-loss penalty multiplies a separately weighted score (move_search.py:289). Mixing is centered and capped (590-603), so percent/cap are coupled and often saturated. Current target bonus and switch margins jointly discourage switches. |

## Risks and Initial Freeze Policy

- Freeze all 12 inactive weights: changes cannot affect current decision flow and would create false optimizer dimensions.
- Freeze all `iterative_search_*` and `deep_extra_depth_margin`: time/depth controls change work and completed depth, not just preferences. Active-game budget tiers (snake_brain.py:403-411) also switch quick scoring at <=18 ms (move_search.py:505). Weight percent and cap change arbitration against a separate fixed evaluator.
- Freeze `position_edge1_penalty` and `position_edge2_penalty`: both score and branch-threshold roles. Freeze advantage activation controls, sequence hold/slack controls, endgame risk window, override margins and territory threshold/caps; discrete branches are not ordinary scalar scoring coefficients. Preserve advantage_min_lead <= advantage_max_activation_lead and min apply depth <= max search depth if later considered.
- Freeze safety, wall/wrong-food/terminal-loss penalties and race-loss penalties initially. `acceptable_risk_bonus` is actually -62; do not silently restore its default +700 or flip its sign. Active `lost_race_penalty` 39205 and `dangerous_penalty` 14806 differ substantially from defaults. Rules remain distinct from these heuristic costs.
- Historical ranges have boundary concentration: food_distance_cost 899/900, lost_race_penalty 39205/40000, position_edge2_penalty 2016/2200; fallback_progress_value equals maximum 2200, one_exit_penalty equals minimum 1000. No evidence establishes those bounds as optimal or safe. All 65 current values are inside historical bounds.
- Existing random_candidate samples all supplied ranges uniformly (tools/optimize_weights.py:84-88). Existing mutate_candidate independently touches each supplied key with 65% probability and scales by +/-30% (91-98;211): roughly 42.25 dimensions per mutation for all 65 ranges. Multiplicative mutation preserves sign absent clamping and cannot move zero values. These mechanics are broader than the proposed local, one-coordinate experiment; constrain tools only in a separate implementation task.
- `search_engine.py` never reads bot weights. Its evaluator uses hardcoded score/region/mobility/center coefficients (22/70/1800/120;451-454), multiplier scaling and no-move adjustments (459-464); terminal priorities at 424-437 are also fixed. `objective_point_value` and similarly valued defaults do not tune those constants. The live iterative_* weights control engine construction and incorporation of its result via move_search.py:554-603.

## Complete Inventory

Paths in the table and JSON are relative to the root in metadata. Existing bounds null means no TUNABLE_RANGES entry, not an unbounded legal domain. 'unused (dormant)' preserves actual references while identifying no current decision effect. Fixed recommended bounds [current,current] mean freeze, not a request to rewrite any weight file.

| Weight | Effective current | Default | Explicit active | Existing min/max | Decision status | Recommendation min/max | Core references |
|---|---:|---:|---|---|---|---|---|
| food_base_value | 10000 | 10000 | yes | none | decision-active | [10000,10000] | snake_bot/food_planner.py:905 |
| objective_point_value | 22 | 22 | no | none | decision-active | [22,22] | snake_bot/food_planner.py:906 |
| multiplier_base_value | 1800 | 1800 | no | none | decision-active | [1800,1800] | snake_bot/food_planner.py:916 |
| multiplier_future_point_value | 7 | 7 | no | none | decision-active | [7,7] | snake_bot/food_planner.py:915 |
| multiplier_denial_point_value | 5 | 5 | no | none | decision-active | [5,5] | snake_bot/food_planner.py:921 |
| multiplier_hunger_penalty | 650 | 650 | no | none | decision-active | [650,650] | snake_bot/food_planner.py:924 |
| multiplier_stack_penalty | 1200 | 1200 | no | none | decision-active | [1200,1200] | snake_bot/food_planner.py:926 |
| multiplier_gap_value | 4500 | 4500 | no | none | decision-active | [4500,4500] | snake_bot/food_planner.py:918 |
| economic_projection_value | 3 | 3 | no | none | decision-active | [3,3] | snake_bot/food_planner.py:907, snake_bot/food_planner.py:1053 |
| economic_projection_cap | 60000 | 60000 | no | none | decision-active | [60000,60000] | snake_bot/food_planner.py:908, snake_bot/food_planner.py:1054 |
| rival_reward_point_value | 5 | 5 | no | none | decision-active | [5,5] | snake_bot/food_planner.py:357 |
| numbered_food_hunger_value | 500 | 500 | no | none | decision-active | [500,500] | snake_bot/food_planner.py:928 |
| sequence_hold_max_remaining | 30 | 30 | no | none | decision-active | [30,30] | snake_bot/food_planner.py:757 |
| sequence_hold_min_enemy_distance | 3 | 3 | no | none | decision-active | [3,3] | snake_bot/food_planner.py:766 |
| sequence_hold_score_buffer | 200 | 200 | no | none | decision-active | [200,200] | snake_bot/food_planner.py:787 |
| food_distance_cost | 899 | 550 | yes | [350,900] | decision-active | [810,988] | snake_bot/food_planner.py:929, snake_bot/food_planner.py:1047 |
| race_margin_value | 332 | 700 | yes | [300,1200] | decision-active | [299,365] | snake_bot/food_planner.py:930, snake_bot/food_planner.py:1046 |
| hunger_value | 233 | 220 | yes | [80,420] | decision-active | [210,256] | snake_bot/food_planner.py:931 |
| space_after_value | 12 | 16 | yes | [6,35] | decision-active | [11,13] | snake_bot/food_planner.py:932, snake_bot/food_planner.py:1049 |
| exits_after_value | 459 | 260 | yes | [80,700] | decision-active | [414,504] | snake_bot/food_planner.py:933, snake_bot/food_planner.py:1050 |
| tail_reachable_bonus | 190 | 450 | yes | [0,1400] | decision-active | [171,209] | snake_bot/food_planner.py:934, snake_bot/food_planner.py:1051 |
| center_unit_value | 53 | 80 | yes | [20,180] | decision-active | [48,58] | snake_bot/food_planner.py:599 |
| edge_depth_value | 463 | 450 | yes | [80,1000] | decision-active | [417,509] | snake_bot/food_planner.py:601 |
| center_edge0_penalty | 4357 | 3200 | yes | [800,8000] | decision-active | [4357,4357] | snake_bot/food_planner.py:603 |
| center_edge1_penalty | 1727 | 1000 | yes | [0,3500] | decision-active | [1727,1727] | snake_bot/food_planner.py:605 |
| next_food_value | 326 | 170 | yes | [40,450] | decision-active | [294,358] | snake_bot/food_planner.py:937 |
| chain_food_value | 260 | 260 | no | [80,650] | decision-active | [234,286] | snake_bot/food_planner.py:284, snake_bot/food_planner.py:311 |
| chain_race_value | 180 | 180 | no | [40,450] | decision-active | [162,198] | snake_bot/food_planner.py:285, snake_bot/food_planner.py:312 |
| chain_close_food_bonus | 900 | 900 | no | [0,2200] | decision-active | [810,990] | snake_bot/food_planner.py:287, snake_bot/food_planner.py:314 |
| sequence_reward_point_value | 4 | 4 | no | none | decision-active | [4,4] | snake_bot/food_planner.py:283, snake_bot/food_planner.py:1073 |
| sequence_setup_base_value | 9000 | 9000 | no | none | decision-active | [9000,9000] | snake_bot/food_planner.py:1043 |
| sequence_setup_reward_point_value | 18 | 18 | no | none | decision-active | [18,18] | snake_bot/food_planner.py:1044 |
| sequence_setup_readiness_value | 1200 | 1200 | no | none | decision-active | [1200,1200] | snake_bot/food_planner.py:1045 |
| sequence_setup_rank_penalty | 4000 | 4000 | no | none | decision-active | [4000,4000] | snake_bot/food_planner.py:1048 |
| sequence_setup_late_slack | 3 | 3 | no | none | decision-active | [3,3] | snake_bot/food_planner.py:1027 |
| sequence_setup_hunger_value | 200 | 200 | no | none | decision-active | [200,200] | snake_bot/food_planner.py:1058 |
| hunger_close_food_value | 90 | 90 | no | [20,220] | decision-active | [81,99] | snake_bot/food_planner.py:289, snake_bot/food_planner.py:316, snake_bot/food_planner.py:941 |
| denial_food_bonus | 1200 | 1200 | no | [300,3600] | decision-active | [1200,1200] | snake_bot/food_planner.py:372, snake_bot/food_planner.py:1153 |
| denial_chain_bonus | 650 | 650 | no | [100,1800] | decision-active | [650,650] | snake_bot/food_planner.py:368, snake_bot/food_planner.py:374, snake_bot/food_planner.py:375, snake_bot/food_planner.py:1153 |
| rival_food_threat_value | 360 | 360 | no | [100,1000] | decision-active | [360,360] | snake_bot/food_planner.py:355, snake_bot/food_planner.py:359, snake_bot/food_planner.py:372 |
| rival_chain_threat_value | 260 | 260 | no | [60,700] | decision-active | [260,260] | snake_bot/food_planner.py:343 |
| rival_chain_ignore_penalty | 1200 | 1200 | no | [400,4000] | decision-active | [1200,1200] | snake_bot/food_planner.py:961 |
| advantage_control_enabled | 1 | 1 | no | none | decision-active | [1,1] | snake_bot/food_planner.py:415 |
| advantage_min_lead | 1600 | 1600 | no | [1200,2600] | decision-active | [1600,1600] | snake_bot/food_planner.py:410 |
| advantage_max_activation_lead | 3000 | 3000 | no | [1800,4000] | decision-active | [3000,3000] | snake_bot/food_planner.py:412 |
| advantage_threat_ratio | 75 | 75 | no | [40,120] | decision-active | [75,75] | snake_bot/food_planner.py:406 |
| advantage_score_buffer | 1000 | 1000 | no | [500,1600] | decision-active | [1000,1000] | snake_bot/food_planner.py:407 |
| advantage_min_body | 8 | 8 | no | [7,12] | decision-active | [8,8] | snake_bot/food_planner.py:418 |
| advantage_body_deficit | 2 | 2 | no | [0,4] | decision-active | [2,2] | snake_bot/food_planner.py:419 |
| advantage_min_own_turns | 4 | 4 | no | none | decision-active | [4,4] | snake_bot/food_planner.py:420 |
| advantage_enemy_cell_value | 150 | 150 | no | [50,350] | decision-active | [150,150] | snake_bot/food_planner.py:492 |
| advantage_our_cell_value | 30 | 30 | no | [0,120] | decision-active | [30,30] | snake_bot/food_planner.py:491 |
| advantage_enemy_move_reduction | 1800 | 1800 | no | [500,5000] | decision-active | [1800,1800] | snake_bot/food_planner.py:493 |
| advantage_food_delay_value | 2200 | 2200 | no | [500,5000] | decision-active | [2200,2200] | snake_bot/food_planner.py:494 |
| advantage_copy_denial_value | 1600 | 1600 | no | none | decision-active | [1600,1600] | snake_bot/food_planner.py:495 |
| advantage_food_unreachable_bonus | 12000 | 12000 | no | [3000,30000] | decision-active | [12000,12000] | snake_bot/food_planner.py:497 |
| advantage_enemy_food_penalty | 20000 | 20000 | no | [5000,45000] | decision-active | [20000,20000] | snake_bot/food_planner.py:499 |
| advantage_switch_margin | 4500 | 4500 | no | [2500,12000] | decision-active | [4500,4500] | snake_bot/food_planner.py:537 |
| advantage_adjacent_food_switch_margin | 15000 | 15000 | no | [8000,30000] | decision-active | [15000,15000] | snake_bot/food_planner.py:539 |
| current_target_bonus | 832 | 500 | yes | [0,3200] | decision-active | [749,915] | snake_bot/food_planner.py:943, snake_bot/food_planner.py:1057, snake_bot/move_search.py:288 |
| safe_bonus | 1680 | 2000 | yes | [700,4000] | decision-active | [1680,1680] | snake_bot/food_planner.py:945 |
| acceptable_risk_bonus | -62 | 700 | yes | [-500,1800] | decision-active | [-62,-62] | snake_bot/food_planner.py:947 |
| dangerous_penalty | 14806 | 6500 | yes | [2500,16000] | decision-active | [14806,14806] | snake_bot/food_planner.py:952 |
| lost_race_penalty | 39205 | 18000 | yes | [6000,40000] | decision-active | [39205,39205] | snake_bot/food_planner.py:959 |
| non_immediate_edge_food_penalty | 5000 | 5000 | no | [1500,14000] | decision-active | [5000,5000] | snake_bot/food_planner.py:653 |
| low_space_food_penalty | 14000 | 14000 | no | [6000,28000] | decision-active | [14000,14000] | snake_bot/food_planner.py:655, snake_bot/food_planner.py:658 |
| long_low_space_food_penalty | 10000 | 10000 | no | [3000,24000] | decision-active | [10000,10000] | snake_bot/food_planner.py:660 |
| one_exit_food_penalty | 8000 | 8000 | no | [3000,18000] | decision-active | [8000,8000] | snake_bot/food_planner.py:663, snake_bot/food_planner.py:665 |
| position_edge0_penalty | 6278 | 6500 | yes | [1500,14000] | decision-active | [6278,6278] | snake_bot/food_planner.py:845 |
| position_edge1_penalty | 3687 | 1800 | yes | [0,5000] | decision-active | [3687,3687] | snake_bot/food_planner.py:847, snake_bot/snake_brain.py:332, snake_bot/snake_brain.py:364 |
| position_edge2_penalty | 2016 | 600 | yes | [0,2200] | decision-active | [2016,2016] | snake_bot/food_planner.py:849, snake_bot/snake_brain.py:319 |
| position_corner_penalty | 206 | 2500 | yes | [0,9000] | decision-active | [206,206] | snake_bot/food_planner.py:860 |
| zero_escape_penalty | 12049 | 9000 | yes | [5000,20000] | decision-active | [12049,12049] | snake_bot/food_planner.py:1401 |
| delayed_zero_penalty | 34313 | 18000 | yes | [9000,40000] | decision-active | [34313,34313] | snake_bot/food_planner.py:1402 |
| one_escape_penalty | 2563 | 2500 | yes | [800,7000] | decision-active | [2563,2563] | snake_bot/food_planner.py:1403 |
| small_region_penalty | 1584 | 1400 | yes | [600,3500] | decision-active | [1584,1584] | snake_bot/food_planner.py:1425, snake_bot/food_planner.py:1450 |
| one_exit_penalty | 1000 | 3500 | yes | [1000,9000] | decision-active | [1000,1000] | snake_bot/food_planner.py:1427, snake_bot/food_planner.py:1452 |
| enemy_no_reply_threat | 80000 | 80000 | yes | none | decision-active | [80000,80000] | snake_bot/food_planner.py:1448 |
| territory_gap_limit | 35 | 35 | yes | none | decision-active | [35,35] | snake_bot/food_planner.py:1454, snake_bot/food_planner.py:1457 |
| territory_gap_penalty | 219 | 250 | yes | [80,600] | decision-active | [219,219] | snake_bot/food_planner.py:1457 |
| territory_gap_penalty_cap | 25000 | 25000 | yes | none | decision-active | [25000,25000] | snake_bot/food_planner.py:1456 |
| territory_control_value | 35 | 35 | no | none | decision-active | [35,35] | snake_bot/food_planner.py:639 |
| territory_control_cap | 7000 | 7000 | no | none | decision-active | [7000,7000] | snake_bot/food_planner.py:640 |
| food_plan_switch_margin | 1200 | 1200 | no | none | decision-active | [1200,1200] | snake_bot/food_planner.py:1139, snake_bot/food_planner.py:1184 |
| food_plan_big_switch_margin | 5000 | 5000 | no | none | decision-active | [5000,5000] | snake_bot/food_planner.py:1191 |
| plan_tactical_override_margin | 1800 | 1800 | no | none | decision-active | [1800,1800] | snake_bot/snake_brain.py:361 |
| adjacent_plan_tactical_override_margin | 10000 | 10000 | no | none | unused | [10000,10000] | none |
| one_escape_tactical_override_margin | 450 | 450 | no | none | decision-active | [450,450] | snake_bot/snake_brain.py:363 |
| edge_tactical_override_margin | 700 | 700 | no | none | decision-active | [700,700] | snake_bot/snake_brain.py:365 |
| survival_region_value | 80 | 80 | no | none | unused (dormant) | [80,80] | snake_bot/food_planner.py:1548 |
| survival_exit_value | 2200 | 2200 | no | none | unused (dormant) | [2200,2200] | snake_bot/food_planner.py:1549 |
| survival_reply_value | 1400 | 1400 | no | none | unused (dormant) | [1400,1400] | snake_bot/food_planner.py:1550 |
| survival_tail_bonus | 1200 | 1200 | no | none | unused (dormant) | [1200,1200] | snake_bot/food_planner.py:1554 |
| survival_immediate_food_bonus | 6500 | 6500 | no | none | unused (dormant) | [6500,6500] | snake_bot/food_planner.py:1627, snake_bot/food_planner.py:1646 |
| enemy_trapped_future_bonus | 220000 | 220000 | no | none | unused (dormant) | [220000,220000] | snake_bot/food_planner.py:1534, snake_bot/food_planner.py:1597 |
| future_no_reply_penalty | 160000 | 160000 | no | none | unused (dormant) | [160000,160000] | snake_bot/food_planner.py:1535, snake_bot/food_planner.py:1561, snake_bot/food_planner.py:1598, snake_bot/food_planner.py:1641, snake_bot/food_planner.py:1660 |
| future_one_reply_penalty | 28000 | 28000 | no | none | unused (dormant) | [28000,28000] | snake_bot/food_planner.py:1563 |
| future_corridor_penalty | 12000 | 12000 | no | none | unused (dormant) | [12000,12000] | snake_bot/food_planner.py:1558, snake_bot/food_planner.py:1565 |
| future_small_region_penalty | 1800 | 1800 | no | none | unused (dormant) | [1800,1800] | snake_bot/food_planner.py:1569 |
| future_enemy_food_penalty | 4500 | 4500 | no | none | unused (dormant) | [4500,4500] | snake_bot/food_planner.py:1653 |
| endgame_food_risk_window | 2 | 2 | no | none | decision-active | [2,2] | snake_bot/food_planner.py:238 |
| endgame_food_bonus | 35000 | 35000 | no | none | decision-active | [35000,35000] | snake_bot/food_planner.py:950, snake_bot/food_planner.py:955, snake_bot/food_planner.py:963 |
| voronoi_cell_value | 95 | 95 | no | none | decision-active | [95,95] | snake_bot/move_search.py:189 |
| voronoi_center_value | 45 | 45 | no | none | decision-active | [45,45] | snake_bot/move_search.py:190 |
| voronoi_food_value | 4200 | 4200 | no | none | decision-active | [4200,4200] | snake_bot/move_search.py:191 |
| voronoi_frontier_penalty | 80 | 80 | no | none | decision-active | [80,80] | snake_bot/move_search.py:192 |
| deep_region_value | 55 | 55 | no | none | decision-active | [55,55] | snake_bot/move_search.py:221, snake_bot/move_search.py:256 |
| deep_exit_value | 1900 | 1900 | no | none | decision-active | [1900,1900] | snake_bot/move_search.py:222, snake_bot/move_search.py:257 |
| deep_reply_value | 1200 | 1200 | no | none | decision-active | [1200,1200] | snake_bot/move_search.py:223, snake_bot/move_search.py:258 |
| deep_tail_bonus | 900 | 900 | no | none | decision-active | [900,900] | snake_bot/move_search.py:226, snake_bot/move_search.py:261 |
| deep_food_now_bonus | 9000 | 9000 | no | none | decision-active | [9000,9000] | snake_bot/move_search.py:294, snake_bot/move_search.py:322 |
| deep_pickup_now_bonus | 8500 | 8500 | no | none | decision-active | [8500,8500] | snake_bot/move_search.py:297, snake_bot/move_search.py:325 |
| deep_score_point_value | 18 | 18 | no | none | decision-active | [18,18] | snake_bot/move_search.py:295, snake_bot/move_search.py:323, snake_bot/move_search.py:368, snake_bot/move_search.py:433 |
| deep_wrong_food_penalty | 140000 | 140000 | no | none | decision-active | [140000,140000] | snake_bot/move_search.py:299, snake_bot/move_search.py:327 |
| deep_wall_hit_penalty | 160000 | 160000 | no | none | decision-active | [160000,160000] | snake_bot/move_search.py:301, snake_bot/move_search.py:329 |
| deep_enemy_wall_hit_bonus | 12000 | 12000 | no | none | decision-active | [12000,12000] | snake_bot/move_search.py:372, snake_bot/move_search.py:437 |
| deep_enemy_trapped_bonus | 120000 | 120000 | no | none | decision-active | [120000,120000] | snake_bot/move_search.py:136 |
| deep_enemy_food_penalty | 6500 | 6500 | no | none | decision-active | [6500,6500] | snake_bot/move_search.py:366, snake_bot/move_search.py:431 |
| deep_enemy_pickup_penalty | 14000 | 14000 | no | none | decision-active | [14000,14000] | snake_bot/move_search.py:370, snake_bot/move_search.py:435 |
| deep_no_reply_penalty | 180000 | 180000 | no | none | decision-active | [180000,180000] | snake_bot/move_search.py:138, snake_bot/move_search.py:232, snake_bot/move_search.py:267, snake_bot/move_search.py:312, snake_bot/move_search.py:471 |
| deep_one_reply_penalty | 30000 | 30000 | no | none | decision-active | [30000,30000] | snake_bot/move_search.py:234, snake_bot/move_search.py:269 |
| deep_risk_penalty | 7000 | 7000 | no | none | decision-active | [7000,7000] | snake_bot/move_search.py:230, snake_bot/move_search.py:265 |
| deep_danger_penalty | 20000 | 20000 | no | none | decision-active | [20000,20000] | snake_bot/move_search.py:228, snake_bot/move_search.py:263 |
| deep_small_region_penalty | 1400 | 1400 | no | none | decision-active | [1400,1400] | snake_bot/move_search.py:237 |
| deep_plan_override_margin | 2600 | 2600 | no | none | decision-active | [2600,2600] | snake_bot/snake_brain.py:336 |
| deep_hot_lost_multiplier | 4 | 4 | no | none | decision-active | [4,4] | snake_bot/move_search.py:289 |
| deep_safe_preference_margin | 30000 | 30000 | no | none | decision-active | [30000,30000] | snake_bot/move_search.py:620 |
| deep_lost_plan_step_penalty | 42000 | 42000 | no | none | decision-active | [42000,42000] | snake_bot/move_search.py:303 |
| deep_extra_depth_margin | 2200 | 2200 | no | none | decision-active | [2200,2200] | snake_bot/move_search.py:539 |
| iterative_search_time_ms | 45 | 45 | no | none | decision-active | [45,45] | snake_bot/move_search.py:556, snake_bot/snake_brain.py:404 |
| iterative_search_max_depth | 6 | 6 | no | none | decision-active | [6,6] | snake_bot/move_search.py:560 |
| iterative_search_min_apply_depth | 4 | 4 | no | none | decision-active | [4,4] | snake_bot/move_search.py:575 |
| iterative_search_weight_percent | 35 | 35 | no | none | decision-active | [35,35] | snake_bot/move_search.py:597 |
| iterative_search_adjustment_cap | 25000 | 25000 | no | none | decision-active | [25000,25000] | snake_bot/move_search.py:600 |
| hot_lost_food_base | 2206 | 2500 | yes | [500,6000] | decision-active | [2206,2206] | snake_bot/food_planner.py:1495, snake_bot/food_planner.py:1517 |
| hot_lost_food_enemy_bonus | 1809 | 1200 | yes | [300,3000] | decision-active | [1809,1809] | snake_bot/food_planner.py:1495, snake_bot/food_planner.py:1517 |
| fallback_progress_value | 2200 | 1000 | yes | [300,2200] | decision-active | [2200,2200] | snake_bot/food_planner.py:1725, snake_bot/move_search.py:285 |
| fallback_hunger_progress_value | 136 | 80 | yes | [20,180] | decision-active | [136,136] | snake_bot/food_planner.py:1725, snake_bot/move_search.py:285 |
| fallback_food_distance_cost | 173 | 80 | yes | [20,220] | decision-active | [173,173] | snake_bot/food_planner.py:1716, snake_bot/move_search.py:286 |
| fallback_hunger_food_progress | 120 | 120 | no | [40,320] | decision-active | [120,120] | snake_bot/food_planner.py:1288 |
| fallback_hunger_food_distance | 55 | 55 | no | [10,160] | decision-active | [55,55] | snake_bot/food_planner.py:1289 |
| fallback_hunger_food_race | 90 | 90 | no | [20,240] | decision-active | [90,90] | snake_bot/food_planner.py:1290 |
| fallback_denial_progress | 120 | 120 | no | [30,450] | decision-active | [108,132] | snake_bot/food_planner.py:1319, snake_bot/food_planner.py:1334 |
| fallback_denial_reach_bonus | 500 | 500 | no | [100,1800] | decision-active | [500,500] | snake_bot/food_planner.py:1319, snake_bot/food_planner.py:1334 |
| fallback_cycle_penalty | 376 | 500 | yes | [100,1500] | decision-active | [339,413] | snake_bot/food_planner.py:1717 |
| fallback_danger_penalty | 11346 | 10000 | yes | [3000,20000] | decision-active | [11346,11346] | snake_bot/food_planner.py:1556, snake_bot/food_planner.py:1704 |
| fallback_wall_hit_penalty | 160000 | 160000 | no | none | decision-active | [160000,160000] | snake_bot/food_planner.py:1735 |

## Machine-Readable Inventory

```json
{"columns":["weight","current","default","explicit_active","existing_bounds","uses","status","reason","recommendation","recommended_bounds"],"rows":[["food_base_value",10000,10000,true,null,["snake_bot/food_planner.py:905"],"decision-active",null,"freeze",[10000,10000]],["objective_point_value",22,22,false,null,["snake_bot/food_planner.py:906"],"decision-active",null,"freeze",[22,22]],["multiplier_base_value",1800,1800,false,null,["snake_bot/food_planner.py:916"],"decision-active",null,"freeze",[1800,1800]],["multiplier_future_point_value",7,7,false,null,["snake_bot/food_planner.py:915"],"decision-active",null,"freeze",[7,7]],["multiplier_denial_point_value",5,5,false,null,["snake_bot/food_planner.py:921"],"decision-active",null,"freeze",[5,5]],["multiplier_hunger_penalty",650,650,false,null,["snake_bot/food_planner.py:924"],"decision-active",null,"freeze",[650,650]],["multiplier_stack_penalty",1200,1200,false,null,["snake_bot/food_planner.py:926"],"decision-active",null,"freeze",[1200,1200]],["multiplier_gap_value",4500,4500,false,null,["snake_bot/food_planner.py:918"],"decision-active",null,"freeze",[4500,4500]],["economic_projection_value",3,3,false,null,["snake_bot/food_planner.py:907","snake_bot/food_planner.py:1053"],"decision-active",null,"freeze",[3,3]],["economic_projection_cap",60000,60000,false,null,["snake_bot/food_planner.py:908","snake_bot/food_planner.py:1054"],"decision-active",null,"freeze",[60000,60000]],["rival_reward_point_value",5,5,false,null,["snake_bot/food_planner.py:357"],"decision-active",null,"freeze",[5,5]],["numbered_food_hunger_value",500,500,false,null,["snake_bot/food_planner.py:928"],"decision-active",null,"freeze",[500,500]],["sequence_hold_max_remaining",30,30,false,null,["snake_bot/food_planner.py:757"],"decision-active",null,"freeze",[30,30]],["sequence_hold_min_enemy_distance",3,3,false,null,["snake_bot/food_planner.py:766"],"decision-active",null,"freeze",[3,3]],["sequence_hold_score_buffer",200,200,false,null,["snake_bot/food_planner.py:787"],"decision-active",null,"freeze",[200,200]],["food_distance_cost",899,550,true,[350,900],["snake_bot/food_planner.py:929","snake_bot/food_planner.py:1047"],"decision-active",null,"perturb",[810,988]],["race_margin_value",332,700,true,[300,1200],["snake_bot/food_planner.py:930","snake_bot/food_planner.py:1046"],"decision-active",null,"perturb",[299,365]],["hunger_value",233,220,true,[80,420],["snake_bot/food_planner.py:931"],"decision-active",null,"perturb",[210,256]],["space_after_value",12,16,true,[6,35],["snake_bot/food_planner.py:932","snake_bot/food_planner.py:1049"],"decision-active",null,"perturb",[11,13]],["exits_after_value",459,260,true,[80,700],["snake_bot/food_planner.py:933","snake_bot/food_planner.py:1050"],"decision-active",null,"perturb",[414,504]],["tail_reachable_bonus",190,450,true,[0,1400],["snake_bot/food_planner.py:934","snake_bot/food_planner.py:1051"],"decision-active",null,"perturb",[171,209]],["center_unit_value",53,80,true,[20,180],["snake_bot/food_planner.py:599"],"decision-active",null,"perturb",[48,58]],["edge_depth_value",463,450,true,[80,1000],["snake_bot/food_planner.py:601"],"decision-active",null,"perturb",[417,509]],["center_edge0_penalty",4357,3200,true,[800,8000],["snake_bot/food_planner.py:603"],"decision-active",null,"freeze",[4357,4357]],["center_edge1_penalty",1727,1000,true,[0,3500],["snake_bot/food_planner.py:605"],"decision-active",null,"freeze",[1727,1727]],["next_food_value",326,170,true,[40,450],["snake_bot/food_planner.py:937"],"decision-active",null,"perturb",[294,358]],["chain_food_value",260,260,false,[80,650],["snake_bot/food_planner.py:284","snake_bot/food_planner.py:311"],"decision-active",null,"perturb",[234,286]],["chain_race_value",180,180,false,[40,450],["snake_bot/food_planner.py:285","snake_bot/food_planner.py:312"],"decision-active",null,"perturb",[162,198]],["chain_close_food_bonus",900,900,false,[0,2200],["snake_bot/food_planner.py:287","snake_bot/food_planner.py:314"],"decision-active",null,"perturb",[810,990]],["sequence_reward_point_value",4,4,false,null,["snake_bot/food_planner.py:283","snake_bot/food_planner.py:1073"],"decision-active",null,"freeze",[4,4]],["sequence_setup_base_value",9000,9000,false,null,["snake_bot/food_planner.py:1043"],"decision-active",null,"freeze",[9000,9000]],["sequence_setup_reward_point_value",18,18,false,null,["snake_bot/food_planner.py:1044"],"decision-active",null,"freeze",[18,18]],["sequence_setup_readiness_value",1200,1200,false,null,["snake_bot/food_planner.py:1045"],"decision-active",null,"freeze",[1200,1200]],["sequence_setup_rank_penalty",4000,4000,false,null,["snake_bot/food_planner.py:1048"],"decision-active",null,"freeze",[4000,4000]],["sequence_setup_late_slack",3,3,false,null,["snake_bot/food_planner.py:1027"],"decision-active",null,"freeze",[3,3]],["sequence_setup_hunger_value",200,200,false,null,["snake_bot/food_planner.py:1058"],"decision-active",null,"freeze",[200,200]],["hunger_close_food_value",90,90,false,[20,220],["snake_bot/food_planner.py:289","snake_bot/food_planner.py:316","snake_bot/food_planner.py:941"],"decision-active",null,"perturb",[81,99]],["denial_food_bonus",1200,1200,false,[300,3600],["snake_bot/food_planner.py:372","snake_bot/food_planner.py:1153"],"decision-active",null,"freeze",[1200,1200]],["denial_chain_bonus",650,650,false,[100,1800],["snake_bot/food_planner.py:368","snake_bot/food_planner.py:374","snake_bot/food_planner.py:375","snake_bot/food_planner.py:1153"],"decision-active",null,"freeze",[650,650]],["rival_food_threat_value",360,360,false,[100,1000],["snake_bot/food_planner.py:355","snake_bot/food_planner.py:359","snake_bot/food_planner.py:372"],"decision-active",null,"freeze",[360,360]],["rival_chain_threat_value",260,260,false,[60,700],["snake_bot/food_planner.py:343"],"decision-active",null,"freeze",[260,260]],["rival_chain_ignore_penalty",1200,1200,false,[400,4000],["snake_bot/food_planner.py:961"],"decision-active",null,"freeze",[1200,1200]],["advantage_control_enabled",1,1,false,null,["snake_bot/food_planner.py:415"],"decision-active",null,"freeze",[1,1]],["advantage_min_lead",1600,1600,false,[1200,2600],["snake_bot/food_planner.py:410"],"decision-active",null,"freeze",[1600,1600]],["advantage_max_activation_lead",3000,3000,false,[1800,4000],["snake_bot/food_planner.py:412"],"decision-active",null,"freeze",[3000,3000]],["advantage_threat_ratio",75,75,false,[40,120],["snake_bot/food_planner.py:406"],"decision-active",null,"freeze",[75,75]],["advantage_score_buffer",1000,1000,false,[500,1600],["snake_bot/food_planner.py:407"],"decision-active",null,"freeze",[1000,1000]],["advantage_min_body",8,8,false,[7,12],["snake_bot/food_planner.py:418"],"decision-active",null,"freeze",[8,8]],["advantage_body_deficit",2,2,false,[0,4],["snake_bot/food_planner.py:419"],"decision-active",null,"freeze",[2,2]],["advantage_min_own_turns",4,4,false,null,["snake_bot/food_planner.py:420"],"decision-active",null,"freeze",[4,4]],["advantage_enemy_cell_value",150,150,false,[50,350],["snake_bot/food_planner.py:492"],"decision-active",null,"freeze",[150,150]],["advantage_our_cell_value",30,30,false,[0,120],["snake_bot/food_planner.py:491"],"decision-active",null,"freeze",[30,30]],["advantage_enemy_move_reduction",1800,1800,false,[500,5000],["snake_bot/food_planner.py:493"],"decision-active",null,"freeze",[1800,1800]],["advantage_food_delay_value",2200,2200,false,[500,5000],["snake_bot/food_planner.py:494"],"decision-active",null,"freeze",[2200,2200]],["advantage_copy_denial_value",1600,1600,false,null,["snake_bot/food_planner.py:495"],"decision-active",null,"freeze",[1600,1600]],["advantage_food_unreachable_bonus",12000,12000,false,[3000,30000],["snake_bot/food_planner.py:497"],"decision-active",null,"freeze",[12000,12000]],["advantage_enemy_food_penalty",20000,20000,false,[5000,45000],["snake_bot/food_planner.py:499"],"decision-active",null,"freeze",[20000,20000]],["advantage_switch_margin",4500,4500,false,[2500,12000],["snake_bot/food_planner.py:537"],"decision-active",null,"freeze",[4500,4500]],["advantage_adjacent_food_switch_margin",15000,15000,false,[8000,30000],["snake_bot/food_planner.py:539"],"decision-active",null,"freeze",[15000,15000]],["current_target_bonus",832,500,true,[0,3200],["snake_bot/food_planner.py:943","snake_bot/food_planner.py:1057","snake_bot/move_search.py:288"],"decision-active",null,"perturb",[749,915]],["safe_bonus",1680,2000,true,[700,4000],["snake_bot/food_planner.py:945"],"decision-active",null,"freeze",[1680,1680]],["acceptable_risk_bonus",-62,700,true,[-500,1800],["snake_bot/food_planner.py:947"],"decision-active",null,"freeze",[-62,-62]],["dangerous_penalty",14806,6500,true,[2500,16000],["snake_bot/food_planner.py:952"],"decision-active",null,"freeze",[14806,14806]],["lost_race_penalty",39205,18000,true,[6000,40000],["snake_bot/food_planner.py:959"],"decision-active",null,"freeze",[39205,39205]],["non_immediate_edge_food_penalty",5000,5000,false,[1500,14000],["snake_bot/food_planner.py:653"],"decision-active",null,"freeze",[5000,5000]],["low_space_food_penalty",14000,14000,false,[6000,28000],["snake_bot/food_planner.py:655","snake_bot/food_planner.py:658"],"decision-active",null,"freeze",[14000,14000]],["long_low_space_food_penalty",10000,10000,false,[3000,24000],["snake_bot/food_planner.py:660"],"decision-active",null,"freeze",[10000,10000]],["one_exit_food_penalty",8000,8000,false,[3000,18000],["snake_bot/food_planner.py:663","snake_bot/food_planner.py:665"],"decision-active",null,"freeze",[8000,8000]],["position_edge0_penalty",6278,6500,true,[1500,14000],["snake_bot/food_planner.py:845"],"decision-active",null,"freeze",[6278,6278]],["position_edge1_penalty",3687,1800,true,[0,5000],["snake_bot/food_planner.py:847","snake_bot/snake_brain.py:332","snake_bot/snake_brain.py:364"],"decision-active",null,"freeze",[3687,3687]],["position_edge2_penalty",2016,600,true,[0,2200],["snake_bot/food_planner.py:849","snake_bot/snake_brain.py:319"],"decision-active",null,"freeze",[2016,2016]],["position_corner_penalty",206,2500,true,[0,9000],["snake_bot/food_planner.py:860"],"decision-active",null,"freeze",[206,206]],["zero_escape_penalty",12049,9000,true,[5000,20000],["snake_bot/food_planner.py:1401"],"decision-active",null,"freeze",[12049,12049]],["delayed_zero_penalty",34313,18000,true,[9000,40000],["snake_bot/food_planner.py:1402"],"decision-active",null,"freeze",[34313,34313]],["one_escape_penalty",2563,2500,true,[800,7000],["snake_bot/food_planner.py:1403"],"decision-active",null,"freeze",[2563,2563]],["small_region_penalty",1584,1400,true,[600,3500],["snake_bot/food_planner.py:1425","snake_bot/food_planner.py:1450"],"decision-active",null,"freeze",[1584,1584]],["one_exit_penalty",1000,3500,true,[1000,9000],["snake_bot/food_planner.py:1427","snake_bot/food_planner.py:1452"],"decision-active",null,"freeze",[1000,1000]],["enemy_no_reply_threat",80000,80000,true,null,["snake_bot/food_planner.py:1448"],"decision-active",null,"freeze",[80000,80000]],["territory_gap_limit",35,35,true,null,["snake_bot/food_planner.py:1454","snake_bot/food_planner.py:1457"],"decision-active",null,"freeze",[35,35]],["territory_gap_penalty",219,250,true,[80,600],["snake_bot/food_planner.py:1457"],"decision-active",null,"freeze",[219,219]],["territory_gap_penalty_cap",25000,25000,true,null,["snake_bot/food_planner.py:1456"],"decision-active",null,"freeze",[25000,25000]],["territory_control_value",35,35,false,null,["snake_bot/food_planner.py:639"],"decision-active",null,"freeze",[35,35]],["territory_control_cap",7000,7000,false,null,["snake_bot/food_planner.py:640"],"decision-active",null,"freeze",[7000,7000]],["food_plan_switch_margin",1200,1200,false,null,["snake_bot/food_planner.py:1139","snake_bot/food_planner.py:1184"],"decision-active",null,"freeze",[1200,1200]],["food_plan_big_switch_margin",5000,5000,false,null,["snake_bot/food_planner.py:1191"],"decision-active",null,"freeze",[5000,5000]],["plan_tactical_override_margin",1800,1800,false,null,["snake_bot/snake_brain.py:361"],"decision-active",null,"freeze",[1800,1800]],["adjacent_plan_tactical_override_margin",10000,10000,false,null,[],"unused","no-core-reference","freeze",[10000,10000]],["one_escape_tactical_override_margin",450,450,false,null,["snake_bot/snake_brain.py:363"],"decision-active",null,"freeze",[450,450]],["edge_tactical_override_margin",700,700,false,null,["snake_bot/snake_brain.py:365"],"decision-active",null,"freeze",[700,700]],["survival_region_value",80,80,false,null,["snake_bot/food_planner.py:1548"],"unused","dormant-survival-call-chain","freeze",[80,80]],["survival_exit_value",2200,2200,false,null,["snake_bot/food_planner.py:1549"],"unused","dormant-survival-call-chain","freeze",[2200,2200]],["survival_reply_value",1400,1400,false,null,["snake_bot/food_planner.py:1550"],"unused","dormant-survival-call-chain","freeze",[1400,1400]],["survival_tail_bonus",1200,1200,false,null,["snake_bot/food_planner.py:1554"],"unused","dormant-survival-call-chain","freeze",[1200,1200]],["survival_immediate_food_bonus",6500,6500,false,null,["snake_bot/food_planner.py:1627","snake_bot/food_planner.py:1646"],"unused","dormant-survival-call-chain","freeze",[6500,6500]],["enemy_trapped_future_bonus",220000,220000,false,null,["snake_bot/food_planner.py:1534","snake_bot/food_planner.py:1597"],"unused","dormant-survival-call-chain","freeze",[220000,220000]],["future_no_reply_penalty",160000,160000,false,null,["snake_bot/food_planner.py:1535","snake_bot/food_planner.py:1561","snake_bot/food_planner.py:1598","snake_bot/food_planner.py:1641","snake_bot/food_planner.py:1660"],"unused","dormant-survival-call-chain","freeze",[160000,160000]],["future_one_reply_penalty",28000,28000,false,null,["snake_bot/food_planner.py:1563"],"unused","dormant-survival-call-chain","freeze",[28000,28000]],["future_corridor_penalty",12000,12000,false,null,["snake_bot/food_planner.py:1558","snake_bot/food_planner.py:1565"],"unused","dormant-survival-call-chain","freeze",[12000,12000]],["future_small_region_penalty",1800,1800,false,null,["snake_bot/food_planner.py:1569"],"unused","dormant-survival-call-chain","freeze",[1800,1800]],["future_enemy_food_penalty",4500,4500,false,null,["snake_bot/food_planner.py:1653"],"unused","dormant-survival-call-chain","freeze",[4500,4500]],["endgame_food_risk_window",2,2,false,null,["snake_bot/food_planner.py:238"],"decision-active",null,"freeze",[2,2]],["endgame_food_bonus",35000,35000,false,null,["snake_bot/food_planner.py:950","snake_bot/food_planner.py:955","snake_bot/food_planner.py:963"],"decision-active",null,"freeze",[35000,35000]],["voronoi_cell_value",95,95,false,null,["snake_bot/move_search.py:189"],"decision-active",null,"freeze",[95,95]],["voronoi_center_value",45,45,false,null,["snake_bot/move_search.py:190"],"decision-active",null,"freeze",[45,45]],["voronoi_food_value",4200,4200,false,null,["snake_bot/move_search.py:191"],"decision-active",null,"freeze",[4200,4200]],["voronoi_frontier_penalty",80,80,false,null,["snake_bot/move_search.py:192"],"decision-active",null,"freeze",[80,80]],["deep_region_value",55,55,false,null,["snake_bot/move_search.py:221","snake_bot/move_search.py:256"],"decision-active",null,"freeze",[55,55]],["deep_exit_value",1900,1900,false,null,["snake_bot/move_search.py:222","snake_bot/move_search.py:257"],"decision-active",null,"freeze",[1900,1900]],["deep_reply_value",1200,1200,false,null,["snake_bot/move_search.py:223","snake_bot/move_search.py:258"],"decision-active",null,"freeze",[1200,1200]],["deep_tail_bonus",900,900,false,null,["snake_bot/move_search.py:226","snake_bot/move_search.py:261"],"decision-active",null,"freeze",[900,900]],["deep_food_now_bonus",9000,9000,false,null,["snake_bot/move_search.py:294","snake_bot/move_search.py:322"],"decision-active",null,"freeze",[9000,9000]],["deep_pickup_now_bonus",8500,8500,false,null,["snake_bot/move_search.py:297","snake_bot/move_search.py:325"],"decision-active",null,"freeze",[8500,8500]],["deep_score_point_value",18,18,false,null,["snake_bot/move_search.py:295","snake_bot/move_search.py:323","snake_bot/move_search.py:368","snake_bot/move_search.py:433"],"decision-active",null,"freeze",[18,18]],["deep_wrong_food_penalty",140000,140000,false,null,["snake_bot/move_search.py:299","snake_bot/move_search.py:327"],"decision-active",null,"freeze",[140000,140000]],["deep_wall_hit_penalty",160000,160000,false,null,["snake_bot/move_search.py:301","snake_bot/move_search.py:329"],"decision-active",null,"freeze",[160000,160000]],["deep_enemy_wall_hit_bonus",12000,12000,false,null,["snake_bot/move_search.py:372","snake_bot/move_search.py:437"],"decision-active",null,"freeze",[12000,12000]],["deep_enemy_trapped_bonus",120000,120000,false,null,["snake_bot/move_search.py:136"],"decision-active",null,"freeze",[120000,120000]],["deep_enemy_food_penalty",6500,6500,false,null,["snake_bot/move_search.py:366","snake_bot/move_search.py:431"],"decision-active",null,"freeze",[6500,6500]],["deep_enemy_pickup_penalty",14000,14000,false,null,["snake_bot/move_search.py:370","snake_bot/move_search.py:435"],"decision-active",null,"freeze",[14000,14000]],["deep_no_reply_penalty",180000,180000,false,null,["snake_bot/move_search.py:138","snake_bot/move_search.py:232","snake_bot/move_search.py:267","snake_bot/move_search.py:312","snake_bot/move_search.py:471"],"decision-active",null,"freeze",[180000,180000]],["deep_one_reply_penalty",30000,30000,false,null,["snake_bot/move_search.py:234","snake_bot/move_search.py:269"],"decision-active",null,"freeze",[30000,30000]],["deep_risk_penalty",7000,7000,false,null,["snake_bot/move_search.py:230","snake_bot/move_search.py:265"],"decision-active",null,"freeze",[7000,7000]],["deep_danger_penalty",20000,20000,false,null,["snake_bot/move_search.py:228","snake_bot/move_search.py:263"],"decision-active",null,"freeze",[20000,20000]],["deep_small_region_penalty",1400,1400,false,null,["snake_bot/move_search.py:237"],"decision-active",null,"freeze",[1400,1400]],["deep_plan_override_margin",2600,2600,false,null,["snake_bot/snake_brain.py:336"],"decision-active",null,"freeze",[2600,2600]],["deep_hot_lost_multiplier",4,4,false,null,["snake_bot/move_search.py:289"],"decision-active",null,"freeze",[4,4]],["deep_safe_preference_margin",30000,30000,false,null,["snake_bot/move_search.py:620"],"decision-active",null,"freeze",[30000,30000]],["deep_lost_plan_step_penalty",42000,42000,false,null,["snake_bot/move_search.py:303"],"decision-active",null,"freeze",[42000,42000]],["deep_extra_depth_margin",2200,2200,false,null,["snake_bot/move_search.py:539"],"decision-active",null,"freeze",[2200,2200]],["iterative_search_time_ms",45,45,false,null,["snake_bot/move_search.py:556","snake_bot/snake_brain.py:404"],"decision-active",null,"freeze",[45,45]],["iterative_search_max_depth",6,6,false,null,["snake_bot/move_search.py:560"],"decision-active",null,"freeze",[6,6]],["iterative_search_min_apply_depth",4,4,false,null,["snake_bot/move_search.py:575"],"decision-active",null,"freeze",[4,4]],["iterative_search_weight_percent",35,35,false,null,["snake_bot/move_search.py:597"],"decision-active",null,"freeze",[35,35]],["iterative_search_adjustment_cap",25000,25000,false,null,["snake_bot/move_search.py:600"],"decision-active",null,"freeze",[25000,25000]],["hot_lost_food_base",2206,2500,true,[500,6000],["snake_bot/food_planner.py:1495","snake_bot/food_planner.py:1517"],"decision-active",null,"freeze",[2206,2206]],["hot_lost_food_enemy_bonus",1809,1200,true,[300,3000],["snake_bot/food_planner.py:1495","snake_bot/food_planner.py:1517"],"decision-active",null,"freeze",[1809,1809]],["fallback_progress_value",2200,1000,true,[300,2200],["snake_bot/food_planner.py:1725","snake_bot/move_search.py:285"],"decision-active",null,"freeze",[2200,2200]],["fallback_hunger_progress_value",136,80,true,[20,180],["snake_bot/food_planner.py:1725","snake_bot/move_search.py:285"],"decision-active",null,"freeze",[136,136]],["fallback_food_distance_cost",173,80,true,[20,220],["snake_bot/food_planner.py:1716","snake_bot/move_search.py:286"],"decision-active",null,"freeze",[173,173]],["fallback_hunger_food_progress",120,120,false,[40,320],["snake_bot/food_planner.py:1288"],"decision-active",null,"freeze",[120,120]],["fallback_hunger_food_distance",55,55,false,[10,160],["snake_bot/food_planner.py:1289"],"decision-active",null,"freeze",[55,55]],["fallback_hunger_food_race",90,90,false,[20,240],["snake_bot/food_planner.py:1290"],"decision-active",null,"freeze",[90,90]],["fallback_denial_progress",120,120,false,[30,450],["snake_bot/food_planner.py:1319","snake_bot/food_planner.py:1334"],"decision-active",null,"perturb",[108,132]],["fallback_denial_reach_bonus",500,500,false,[100,1800],["snake_bot/food_planner.py:1319","snake_bot/food_planner.py:1334"],"decision-active",null,"freeze",[500,500]],["fallback_cycle_penalty",376,500,true,[100,1500],["snake_bot/food_planner.py:1717"],"decision-active",null,"perturb",[339,413]],["fallback_danger_penalty",11346,10000,true,[3000,20000],["snake_bot/food_planner.py:1556","snake_bot/food_planner.py:1704"],"decision-active",null,"freeze",[11346,11346]],["fallback_wall_hit_penalty",160000,160000,false,null,["snake_bot/food_planner.py:1735"],"decision-active",null,"freeze",[160000,160000]]]}
```
