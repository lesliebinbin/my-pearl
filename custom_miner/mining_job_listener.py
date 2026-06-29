#!/usr/bin/env python3
"""Poll Pearl Gateway for mining jobs and log them."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger("custom_miner.mining_job_listener")


@dataclass(frozen=True)
class GatewayConfig:
    transport: str
    socket_path: str
    host: str
    port: int


class GatewayJsonRpcClient:
    """Newline-delimited JSON-RPC client for Pearl Gateway's Miner RPC."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._request_id = 0
        self._socket = self._connect()
        self._reader = self._socket.makefile("r", encoding="utf-8")
        self._writer = self._socket.makefile("w", encoding="utf-8")

    def _connect(self) -> socket.socket:
        if self._config.transport == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self._config.host, self._config.port))
            LOGGER.info("Connected to gateway at %s:%s", self._config.host, self._config.port)
            return sock

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self._config.socket_path)
        LOGGER.info("Connected to gateway at %s", self._config.socket_path)
        return sock

    def close(self) -> None:
        self._reader.close()
        self._writer.close()
        self._socket.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._request_id,
        }

        self._writer.write(json.dumps(request) + "\n")
        self._writer.flush()

        response_line = self._reader.readline()
        if not response_line:
            raise ConnectionError("gateway closed the connection")

        response = json.loads(response_line)
        if response.get("id") != self._request_id:
            raise RuntimeError(
                f"JSON-RPC response id mismatch: expected {self._request_id}, got {response.get('id')}"
            )

        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"gateway error {error.get('code')}: {error.get('message')}")

        return response.get("result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for mining jobs from Pearl Gateway")
    parser.add_argument(
        "--transport",
        choices=("uds", "tcp"),
        default=os.getenv("MINER_RPC_TRANSPORT", "uds"),
        help="Gateway transport. Defaults to MINER_RPC_TRANSPORT or uds.",
    )
    parser.add_argument(
        "--socket-path",
        default=os.getenv("MINER_RPC_SOCKET_PATH", "/tmp/pearlgw.sock"),
        help="Gateway Unix socket path. Defaults to MINER_RPC_SOCKET_PATH or /tmp/pearlgw.sock.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MINER_RPC_HOST", "localhost"),
        help="Gateway TCP host. Defaults to MINER_RPC_HOST or localhost.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MINER_RPC_PORT", "8337")),
        help="Gateway TCP port. Defaults to MINER_RPC_PORT or 8337.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between getMiningInfo calls.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch one mining job and exit.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOGGING_LEVEL", "info"),
        choices=("debug", "info", "warning", "error", "critical"),
        help="Log level.",
    )
    return parser.parse_args()


def decode_mining_job(job: dict[str, Any]) -> dict[str, Any]:
    header = base64.b64decode(job["incomplete_header_bytes"])
    return {
        "incomplete_header_bytes": job["incomplete_header_bytes"],
        "incomplete_header_hex": header.hex(),
        "incomplete_header_size": len(header),
        "target": job["target"],
        "cert_version": job.get("cert_version"),
    }


def log_mining_job(job: dict[str, Any]) -> None:
    decoded = decode_mining_job(job)
    LOGGER.info("Mining job: %s", json.dumps(decoded, sort_keys=True))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = GatewayConfig(
        transport=args.transport,
        socket_path=args.socket_path,
        host=args.host,
        port=args.port,
    )

    client = GatewayJsonRpcClient(config)
    try:
        while True:
            log_mining_job(client.call("getMiningInfo"))
            if args.once:
                return
            time.sleep(args.interval)
    finally:
        client.close()


if __name__ == "__main__":
    main()
