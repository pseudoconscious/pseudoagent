"""
Tool handlers for the counter example agent.

Every handler has the same uniform contract:

    handler(data, params, context)

    data    - the shared agent state (mutable dict, any shape)
    params  - dict extracted from the LLM tool call
              (empty dict for tools with no llm_tool schema)
    context - engine services (llm, config, stop signal)

Handlers are plain Python functions: lintable, testable,
importable. They know nothing about AHP or the engine loop;
they just read/write state, and a handler that wants the
agent to stop calls context.stop().

To add a new tool:
    1. write a handler here (or in any importable module)
    2. declare it in counter.yaml (tools.<name>.handler)
    3. optionally give it an llm_tool schema for parameters
"""


def set_counter(data, params, context):
    """
    Set data["counter"] to params["num"].

    The num parameter is not hardcoded: it was extracted from
    an LLM tool call driven by the set_tool schema.
    """

    data["counter"] = int(params["num"])

    return f"SET counter = {data['counter']}"


def increase_counter(data, params, context):
    """
    Increase data["counter"] by exactly 1.
    """

    data["counter"] += 1

    return f"INCREASE counter -> {data['counter']}"


def decrease_counter(data, params, context):
    """
    Decrease data["counter"] by exactly 1.
    """

    data["counter"] -= 1

    return f"DECREASE counter -> {data['counter']}"


def stop(data, params, context):
    """
    Stop the agent loop.

    Some agents are exit-less and simply never register this
    handler; termination policy then lives in the config
    (settings.max_iterations) instead.
    """

    context.stop("mandate fulfilled")


def digest(data, context):
    print("Digest here!")
