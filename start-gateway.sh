#!/usr/bin/env zsh
PEARLD_RPC_URL="http://100.93.219.53:44107" \
PEARLD_RPC_USER="rpcuser" \
PEARLD_RPC_PASSWORD="rpcpass" \
PEARLD_MINING_ADDRESS="prl1pf2uu40cgjrqs0000rgv592fscn65patpgel6yzzlaak6rearpxqqjpegjc" \
LOGGING_LEVEL="debug" \
uv run --package pearl-gateway pearl-gateway start 

