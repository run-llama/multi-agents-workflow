import asyncio
from typing import Any, List

from llama_index.core.tools import FunctionTool

from app.core.planner_agent import StructuredPlannerAgent
from app.core.prefix import PrintPrefix
from app.core.function_call import AgentRunResult, FunctionCallingAgent

import textwrap


class AgentCallingAgent(FunctionCallingAgent):
    def __init__(
        self,
        *args: Any,
        name: str,
        agents: List[FunctionCallingAgent] | None = None,
        **kwargs: Any,
    ) -> None:
        agents = agents or []
        tools = [_create_call_workflow_fn(self, agent) for agent in agents]
        super().__init__(*args, name=name, tools=tools, **kwargs)
        # call add_workflows so agents will get detected by llama agents automatically
        self.add_workflows(**{agent.name: agent for agent in agents})


class AgentOrchestrator(StructuredPlannerAgent):
    def __init__(
        self,
        *args: Any,
        agents: List[FunctionCallingAgent] | None = None,
        name: str = "orchestrator",
        **kwargs: Any,
    ) -> None:
        agents = agents or []
        tools = [_create_call_workflow_fn(self, agent) for agent in agents]
        super().__init__(
            *args,
            name=name,
            tools=tools,
            **kwargs,
        )
        # call add_workflows so agents will get detected by llama agents automatically
        self.add_workflows(**{agent.name: agent for agent in agents})


def _create_call_workflow_fn(
    caller: FunctionCallingAgent, agent: FunctionCallingAgent
) -> FunctionTool:
    def info(prefix: str, text: str) -> None:
        truncated = textwrap.shorten(text, width=255, placeholder="...")
        print(f"{prefix}: '{truncated}'")

    async def acall_workflow_fn(input: str) -> str:
        # info(f"[{caller_name}->{agent.name}]", input)
        task = asyncio.create_task(agent.run(input=input))
        # bubble all events while running the agent to the calling agent
        if len(caller._sessions) > 1:
            print("XXX: Bubbling events only works with single-session agents")
        else:
            session = next(iter(caller._sessions))
            async for ev in agent.stream_events():
                session.write_event_to_stream(ev)
        ret: AgentRunResult = await task
        response = ret.response.message.content
        # info(f"[{caller_name}<-{agent.name}]", response)
        return response

    return FunctionTool.from_defaults(
        async_fn=acall_workflow_fn,
        name=f"call_{agent.name}",
        description=(
            f"Use this tool to delegate a sub task to the {agent.name} agent."
            + (f" The agent is an {agent.role}." if agent.role else "")
        ),
    )
