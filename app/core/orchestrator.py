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
from app.core.planner import Planner, SubTask


class StartPlanEvent(Event):
    plan_id: str


class SubTaskEvent(Event):
    plan_id: str
    sub_task: SubTask


class SubTaskResultEvent(Event):
    plan_id: str
    sub_task: SubTask


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

    @step()
    async def create_plan(self, ev: StartEvent) -> StartPlanEvent | StopEvent:
        plan_id = await self.planner.create_plan(input=ev.input)
        if plan_id is None:
            return StopEvent()
        return StartPlanEvent(plan_id=plan_id)

    @step()
    async def execute_plan(self, ctx: Context, ev: StartPlanEvent) -> SubTaskEvent:
        upcoming_sub_tasks = self.planner.state.get_next_sub_tasks(ev.plan_id)

        if len(upcoming_sub_tasks) == 0:
            return StopEvent(result={"response": ""})

        ctx.data["num_sub_tasks"] = len(upcoming_sub_tasks)

        # send an event per sub task
        events = [
            SubTaskEvent(sub_task=sub_task, plan_id=ev.plan_id)
            for sub_task in upcoming_sub_tasks
        ]
        for event in events:
            ctx.session.send_event(event)

        return None

    @step()
    async def execute_sub_task(self, ev: SubTaskEvent) -> SubTaskResultEvent:
        print(f"Executing sub task: {ev.sub_task.name}")
        self.planner.state.add_completed_sub_task(ev.plan_id, ev.sub_task)
        return SubTaskResultEvent(sub_task=ev.sub_task, plan_id=ev.plan_id)

    @step()
    async def gather_results(self, ctx: Context, ev: SubTaskResultEvent) -> StopEvent:
        # wait for sub tasks to finish
        num_sub_tasks = ctx.data["num_sub_tasks"]
        results = ctx.collect_events(ev, [SubTaskResultEvent] * num_sub_tasks)
        if results is None:
            return None

        # TODO: implement refining the plan

        return StopEvent(result=results)
