# PseudoAgent
A new AI agent paradigm!

It knows *mechanisms*, never tool
names. Everything an agent needs — its state, its tools, its
LLMs, its stopping policy — is declared in a YAML file.

## Library layout

```
pseudoagent/
    pseudoagent.py             stateless agent loop (run, run_tool)
    pseudoagent_ahp_helper.py  AHP tool selection
    pseudoagent_agent_types.py NormalAgent / StopAgent
    pseudoagent_tools_helper.py registry + LLM tool schema machinery
```

The package ships **no configuration and no tools** — an agent
application is your config + your handlers (see below).

## Install

```
pip install -e .
```

## Usage as a library

```python
from pseudoagent import load_config, run

config = load_config("my_agent.yaml")
run(config)
```

The YAML declares the LLM providers, the agent state, the tools
(each pointing at *your* handler functions via dotted paths),
the native LLM tool schemas, the AHP criterias, and the engine
settings. See the example for a complete, commented file.

## Counter example

A complete, minimal application lives in `examples/counter`:

```
examples/counter/
    counter.py        entry point: loads counter.yaml, runs the loop
    counter.yaml      state, tools, LLM setup, AHP criterias
    counter_tools.py  the tool handlers this app registers
```

Run it from that folder:

```
cd examples/counter
python counter.py
```

It drives the same engine you would use from an external
program (like `pseudochat`): the example imports `pseudoagent`
exactly as any installed-package consumer would.
