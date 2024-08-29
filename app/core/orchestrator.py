from typing import Any, List

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.settings import Settings
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from app.core.agent_call import create_call_workflow_fn
from app.core.function_call import FunctionCallingAgent
from app.core.planner import Planner, SubTask


class ExecutePlanEvent(Event):
    pass


class SubTaskEvent(Event):
    sub_task: SubTask


class SubTaskResultEvent(Event):
    sub_task: SubTask
    result: str


class PlanningOrchestrator(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        agents: List[Workflow] | None = None,
        timeout: float = 360.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, timeout=timeout, **kwargs)
        self.name = "orchestrator"

        if llm is None:
            llm = Settings.llm
        self.llm = llm
        assert self.llm.metadata.is_function_calling_model

        self.tools = [create_call_workflow_fn(self.name, agent) for agent in agents]
        self.planner = Planner(tools=self.tools)
        # The executor is keeping the memory of all tool calls and decides to call the right tool for the task
        self.executor = FunctionCallingAgent(
            name="executor",
            tools=self.tools,
            system_prompt="You are an expert in completing given tasks by calling the right tool for the task.",
        )

    @step()
    async def create_plan(
        self, ctx: Context, ev: StartEvent
    ) -> ExecutePlanEvent | StopEvent:
        plan_id = await self.planner.create_plan(input=ev.input)
        ctx.data["act_plan_id"] = plan_id
        print("=== Executing plan ===")
        return ExecutePlanEvent()

    @step()
    async def execute_plan(self, ctx: Context, ev: ExecutePlanEvent) -> SubTaskEvent:
        upcoming_sub_tasks = self.planner.state.get_next_sub_tasks(
            ctx.data["act_plan_id"]
        )

        ctx.data["num_sub_tasks"] = len(upcoming_sub_tasks)
        # send an event per sub task
        events = [SubTaskEvent(sub_task=sub_task) for sub_task in upcoming_sub_tasks]
        for event in events:
            ctx.session.send_event(event)

        return None

    @step()
    async def execute_sub_task(
        self, ctx: Context, ev: SubTaskEvent
    ) -> SubTaskResultEvent:
        print(f"=== Executing sub task: {ev.sub_task.name} ===")
        result = await self.executor.run(input=ev.sub_task.input)
        result = str(result["response"])
        print("=== Done executing sub task ===")
        self.planner.state.add_completed_sub_task(ctx.data["act_plan_id"], ev.sub_task)
        return SubTaskResultEvent(sub_task=ev.sub_task, result=result)

    @step()
    async def gather_results(
        self, ctx: Context, ev: SubTaskResultEvent
    ) -> ExecutePlanEvent | StopEvent:
        # wait for all sub tasks to finish
        num_sub_tasks = ctx.data["num_sub_tasks"]
        results = ctx.collect_events(ev, [SubTaskResultEvent] * num_sub_tasks)
        assert results is not None

        # if no more tasks to do, stop workflow and send result of last step
        upcoming_sub_tasks = self.planner.state.get_next_sub_tasks(
            ctx.data["act_plan_id"]
        )
        if len(upcoming_sub_tasks) == 0:
            return StopEvent(result=results[-1].result)

        # continue executing plan
        return ExecutePlanEvent()
