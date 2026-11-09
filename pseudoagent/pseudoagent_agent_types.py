"""
Shared engine types.

This tiny module exists so that the engine (pseudoagent.py)
and tool handlers can share StopAgent and NormalAgent without
importing each other.
"""


class StopAgent(Exception):
    """
    Raised by a tool handler (via context.stop()) to stop the
    agent loop. Exit-less agents never register a stop handler;
    termination then comes from settings.max_iterations.
    """


class NormalAgent:
    """
    Engine services handed to every tool handler as context.

    llm    - the LLM object (handlers may do LLM work of their own)
    config - the full loaded configuration
    """

    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

    def stop(self, reason="stopped by handler"):
        raise StopAgent(reason)
