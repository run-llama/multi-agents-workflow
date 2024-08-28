from typing import Any, List, Optional

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Workflow

from app.core.prefix import PrintPrefix
from app.core.function_call import FunctionCallingAgent

import textwrap


def create_call_workflow_fn(caller_name: str, agent: Workflow) -> FunctionTool:
    def call_workflow_fn(input: str) -> str:
        raise NotImplementedError

    def info(prefix: str, text: str) -> None:
        truncated = textwrap.shorten(text, width=255, placeholder="...")
        print(f"{prefix}: '{truncated}'")

    async def acall_workflow_fn(input: str) -> str:
        info(f"[{caller_name}->{agent.name}]", input)
        with PrintPrefix(f"[{agent.name}]"):
            ret = await agent.run(input=input)
            response = str(ret["response"])
        info(f"[{caller_name}<-{agent.name}]", response)
        return response

    return FunctionTool.from_defaults(
        fn=call_workflow_fn,  # not necessary with https://github.com/run-llama/llama_index/pull/15638/files
        async_fn=acall_workflow_fn,
        name=f"call_{agent.name}",
        description=(
            f"Use this tool to delegate a sub task to the {agent.name} agent."
            + (f" The agent is an {agent.role}." if agent.role else "")
        ),
    )


class AgentCallingAgent(FunctionCallingAgent):
    def __init__(
        self,
        *args: Any,
        agents: List[Workflow] | None = None,
        **kwargs: Any,
    ) -> None:
        agents = agents or []
        tools = [create_call_workflow_fn(self.name, agent) for agent in agents]
        super().__init__(*args, tools=tools, **kwargs)
        # call add_workflows so agents will get detected by llama agents automatically
        self.add_workflows(**{agent.name: agent for agent in agents})


class AgentOrchestrator(AgentCallingAgent):
    def __init__(
        self,
        *args: Any,
        agents: List[Workflow] | None = None,
        name: str = "orchestrator",
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        system_prompt = (
            system_prompt
            or """You are given a specific task. Don't try to do the task yourself. Don't make up any information yourself. Instead use any of the provided tools to complete the task. Each tool provides an interface for calling a different expert agent.
        First make a plan accordingly to the original task. Then execute the plan by delegating each task of the plan to the different experts (i.e. calling their tools)."""
        )
        super().__init__(
            *args,
            agents=agents,
            name=name,
            system_prompt=system_prompt,
            **kwargs,
        )
