"""
server/app.py — OpenEnv entry point for multi-mode deployment.
This file is required by the OpenEnv spec for multi-mode deployment.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from supply_chain_env.server import app

__all__ = ["app"]
