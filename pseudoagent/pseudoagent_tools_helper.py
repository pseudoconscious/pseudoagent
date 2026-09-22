"""
Tool machinery helpers.

Everything here is pure build/parse logic over dicts and
strings: resolving handlers, building LLM tool schemas,
extracting parameters from raw LLM responses.

This module is a leaf: it imports nothing from the engine
(pseudoagent.py) or from other helpers, so it stays trivially
testable. The engine imports these helpers, never the
reverse.
"""

import importlib
import json


def resolve_handler(dotted_path):
    """
    Import a handler from a dotted path like
    "counter_tools.set_counter" (import, never eval).
    """

    module_path, _, function_name = dotted_path.rpartition(".")

    if not module_path:
        raise RuntimeError(
            f"Handler '{dotted_path}' must be a dotted path "
            "like 'module.function'."
        )

    module = importlib.import_module(module_path)

    handler = getattr(module, function_name, None)

    if not callable(handler):
        raise RuntimeError(
            f"Handler '{dotted_path}' was found but is not callable."
        )

    return handler


def build_registry(config):
    """
    Resolve every tool's handler at startup and fail fast,
    listing all missing ones at once, if any is missing.
    """

    registry = {}
    problems = []

    for name, spec in (config.get("tools") or {}).items():

        handler_path = (spec or {}).get("handler")

        if not handler_path:
            problems.append(f"tool '{name}': no handler declared")
            continue

        try:
            registry[name] = {
                "handler": resolve_handler(handler_path),
                "llm_tool": (spec or {}).get("llm_tool"),
            }
        except Exception as error:
            problems.append(f"tool '{name}': {error}")

    if problems:
        raise RuntimeError(
            "The tool registry failed to build:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )

    return registry


def build_llm_tool_schema(name, spec):
    """
    Turn one llm_tools entry from the config into an
    OpenAI-style function schema for generate_raw(tools=...).

    If "required" is omitted, every declared parameter is
    required.
    """

    properties = {}

    for param_name, param_spec in (spec.get("parameters") or {}).items():

        properties[param_name] = {
            "type": param_spec.get("type", "string"),
            "description": param_spec.get("description", ""),
        }

    required = spec.get("required")

    if required is None:
        required = list(properties)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": spec.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_llm_tools(config):
    """
    Build all native LLM tool schemas from the llm_tools
    section of the config, and fail fast if a tool's declared
    llm_tool schema is missing.
    """

    schemas = {
        name: build_llm_tool_schema(name, spec)
        for name, spec in (config.get("llm_tools") or {}).items()
    }

    for name, spec in (config.get("tools") or {}).items():

        llm_tool = (spec or {}).get("llm_tool")

        if llm_tool and llm_tool not in schemas:
            raise RuntimeError(
                f"Tool '{name}' declares llm_tool '{llm_tool}' "
                "but it is missing from llm_tools."
            )

    return schemas


def extract_tool_call_arguments(response, tool_name):
    """
    Inspect a raw generate_raw response and return the arguments
    of the first tool call for tool_name.

    The generated text (and the reasoning content) is discarded:
    only the tool call JSON is looked at.

    generate_raw returns the full chat.completion response, so
    tool_calls live in choices[0].message.tool_calls. A top-level
    tool_calls key is also checked for tolerance.
    """

    if isinstance(response, dict):

        tool_calls = response.get("tool_calls")

        if not tool_calls:

            choices = response.get("choices") or []

            if choices:
                message = choices[0].get("message") or {}
                tool_calls = message.get("tool_calls")

        tool_calls = tool_calls or []

    elif isinstance(response, list):
        tool_calls = response

    else:
        tool_calls = []

    for call in tool_calls:

        function = call.get("function") or {}

        if function.get("name") != tool_name:
            continue

        arguments = function.get("arguments") or {}

        if isinstance(arguments, str):

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if arguments:
            return arguments

    raise RuntimeError(
        f"The LLM did not return a {tool_name} tool call "
        "with arguments."
    )
