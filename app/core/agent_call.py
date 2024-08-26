from typing import Any, List

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Workflow

from app.core.prefix import PrintPrefix
from app.core.function_call import FunctionCallingAgent


def create_call_workflow_fn(agent: Workflow) -> FunctionTool:
    def call_workflow_fn(input: str) -> str:
        raise NotImplementedError

    async def acall_workflow_fn(input: str) -> str:
        print(f"Calling agent {agent.name} with input: '{input}'")
        with PrintPrefix(f"[{agent.name}]"):
            ret = await agent.run(input=input)
        print(f"Finished calling agent {agent.name}")
        return ret["response"]

    return FunctionTool.from_defaults(
        fn=call_workflow_fn,  # not necessary with https://github.com/run-llama/llama_index/pull/15638/files
        async_fn=acall_workflow_fn,
        name=f"call_{agent.name}",
        description=f"Use this tool to delegate a task to the agent {agent.name}",
    )


class AgentCallingAgent(FunctionCallingAgent):
    def __init__(
        self,
        *args: Any,
        agents: List[Workflow] | None = None,
        **kwargs: Any,
    ) -> None:
        agents = agents or []
        tools = [create_call_workflow_fn(agent) for agent in agents]
        super().__init__(*args, tools=tools, **kwargs)
