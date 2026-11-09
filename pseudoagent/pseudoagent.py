import json
from time import sleep
import yaml
from frdfm_llm_tools import LLM
from .pseudoagent_ahp_helper import choose_ahp
from .pseudoagent_agent_types import NormalAgent, StopAgent
from .pseudoagent_tools_helper import (
    build_registry,
    build_llm_tools,
    extract_tool_call_arguments,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------


def load_config(path):
    """
    Load the agent state and configuration from a YAML file.

    The library ships no config: callers pass the path to
    their own configuration file (see examples/counter).
    """

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_llm(config):
    """
    Build the LLM object from the llm section of the config.
    """

    llm_config = config["llm"]

    return LLM(
        app_title="PseudoAgent",
        app_url="https://www.pseudoconscious.com/pseudoagent",
        app_version="0.1.0",
        generate_provider=llm_config["generate"]["provider"],
        generate_env_path=llm_config["generate"]["env_path"],
        choose_provider=llm_config["choose"]["provider"],
        choose_env_path=llm_config["choose"]["env_path"],
    )


# ---------------------------------------------------------
# Tool execution
# ---------------------------------------------------------

def run_tool(tool, data, registry, llm_tools, llm, context):
    """
    Execute one selected tool through the registry:

      AHP picked the name -> (optional) extract parameters from
      an LLM tool call -> handler(data, params, context).

    Tools without an llm_tool schema run with empty params and
    no LLM call. A handler that wants to stop the agent raises
    StopAgent (context.stop() does this).
    """

    if tool == "digest":
        return

    entry = registry.get(tool)

    if entry is None:
        raise RuntimeError(
            f"Selected tool '{tool}' is not in the registry."
        )

    params = {}

    schema_name = entry.get("llm_tool")

    if schema_name:

        schema = llm_tools[schema_name]
        tool_call_name = schema["function"]["name"]

        prompt = f"""
            You must run the {tool_call_name} tool.
            
            Current data:
            {json.dumps(data, indent=4)}
            
            Run {tool_call_name} with the proper parameters so that the
            mandate in the data can be fulfilled.
            Do not answer with text. Run the tool.
        """

        response = llm.generate_raw(
            prompt,
            tools=[schema],
        )

        params = extract_tool_call_arguments(response, tool_call_name)

    result = entry["handler"](data, params, context)

    print(f"  -> {result}")


def digest(data, context):
    """
    Fallback digest used when the config declares no digest
    tool; run() replaces it with the handler from the registry.
    """
    pass


# ---------------------------------------------------------
# Stateless agent loop
# ---------------------------------------------------------

def run(config):
    """
    Run the agent loop until a handler raises StopAgent or
    settings.max_iterations is reached.
    """

    llm = build_llm(config)
    normal_agent = NormalAgent(llm, config)

    data = config["data"]
    tools = config["tools"]
    criterias = config["criterias"]
    settings = config.get("settings") or {}

    registry = build_registry(config)
    digest = registry['digest']['handler']
    llm_tools = build_llm_tools(config)

    loop_delay_seconds = settings.get("loop_delay_seconds", 1)
    max_iterations = settings.get("max_iterations")

    if max_iterations:
        iterations = range(1, max_iterations + 1)
    else:
        iterations = iter(int, 1)  # count forever

    for iteration in iterations:

        print()
        print(f"### ITERATION {iteration} ###")
        print()

        print(
            json.dumps(
                data,
                indent=4
            )
        )

        tool = choose_ahp(
            llm,
            data,
            tools,
            criterias,
        )

        print()
        print("### SELECTED TOOL ###")
        print(tool)

        try:

            run_tool(
                tool,
                data,
                registry,
                llm_tools,
                llm,
                normal_agent,
            )

            digest(data, normal_agent)

        except StopAgent as reason:

            print()
            print(f"### STOPPED: {reason} ###")
            return

        sleep(loop_delay_seconds)

    print()
    print(f"### Reached max_iterations ({max_iterations}); stopping. ###")
