"""Identify the runtime release written into match metadata and local reports."""

BOT_VERSION = '2.17'

BOT_VERSION_NOTES = (
    'official v11 cube, partial observations, timestamped memory, '
    'six-direction actions and shared game transitions'
)
# python -m tools.bot_lab simulate --games 20 --seed 1000 --turns 150
# python -m tools.bot_lab logs game_*.log --max-positions 500
