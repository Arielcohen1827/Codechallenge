#!/usr/bin/env bash
# Run codechallenge-test-client: a websocket bot that connects to the local
# server (ws://localhost:5000/ws) and auto-plays challenges.
#
# Usage: ./start.sh <auth_token>
#   <auth_token> is your Bot token from the web app (auth_app Bot.token).
#   Requires the server to be up (codechallenge-server on :5000).
set -euo pipefail
cd "$(dirname "$0")"

# Provision the venv on first run (only dependency is websockets).
if [ ! -d .venv ]; then
    echo "Creating .venv and installing dependencies..." >&2
    python3 -m venv .venv
    ./.venv/bin/pip install -q -r requirements.txt
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ "$#" -lt 1 ]; then
    echo "Usage: ./start.sh <auth_token>" >&2
    echo "  Get the token from the web app (your Bot's token)." >&2
    echo "  https://codechallenge.net.ar/mybots " >&2
    exit 1
fi

# run async
# exec python run.py "$1"
# run sync
# python run.py "$1"

python run.py "$1"
