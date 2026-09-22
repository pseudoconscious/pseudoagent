"""
Counter example: a minimal PseudoAgent application.

The pseudoagent package is a stateless engine; this folder is
a complete, self-contained application that drives it:

    counter.py        - entry point: loads counter.yaml, runs the loop
    counter.yaml      - state, tools, LLM setup, AHP criterias
    counter_tools.py  - the tool handlers this app registers

Run from the examples/counter folder:

    python counter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pseudoagent import load_config, run


if __name__ == "__main__":

    config = load_config(Path(__file__).with_name("counter.yaml"))

    run(config)
