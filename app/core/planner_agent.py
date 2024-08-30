from typing import Any, List

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from app.core.function_call import AgentRunResult, FunctionCallingAgent
from app.core.planner import Planner, SubTask, Plan
from llama_index.core.tools import BaseTool

from enum import Enum


class ExecutePlanEvent(Event):
    pass


class SubTaskEvent(Event):
    sub_task: SubTask


class SubTaskResultEvent(Event):
    sub_task: SubTask
    result: AgentRunResult


class PlanEventType(Enum):
    CREATED = "created"
    REFINED = "refined"


class PlanEvent(Event):

    event_type: PlanEventType
    plan: Plan

    @property
    def msg(self) -> str:
        sub_task_names = ", ".join(task.name for task in self.plan.sub_tasks)
        return f"Plan {self.event_type.value}: Let's do: {sub_task_names}"


class StructuredPlannerAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        name: str,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        timeout: float = 360.0,
        refine_plan: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, timeout=timeout, **kwargs)
        self.name = name
        self.refine_plan = refine_plan

        self.tools = tools or []
        self.planner = Planner(llm=llm, tools=self.tools, verbose=self._verbose)
        # The executor is keeping the memory of all tool calls and decides to call the right tool for the task
        self.executor = FunctionCallingAgent(
            name="executor",
            llm=llm,
            tools=self.tools,
            # it's important to instruct to just return the tool call, otherwise the executor will interpret and change the result
            system_prompt="You are an expert in completing given tasks by calling the right tool for the task. Just return the result of the tool call. Don't add any information yourself",
        )

    @step()
    async def create_plan(
        self, ctx: Context, ev: StartEvent
    ) -> ExecutePlanEvent | StopEvent:
        plan_id, plan = await self.planner.create_plan(input=ev.input)
        ctx.data["task"] = ev.input
        ctx.data["act_plan_id"] = plan_id
        # inform about the new plan
        ctx.session.write_event_to_stream(
            PlanEvent(event_type=PlanEventType.CREATED, plan=plan)
        )
        if self._verbose:
            print("=== Executing plan ===\n")
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
        if self._verbose:
            print(f"=== Executing sub task: {ev.sub_task.name} ===")
        result: AgentRunResult = await self.executor.run(input=ev.sub_task.input)
        if self._verbose:
            print("=== Done executing sub task ===\n")
        self.planner.state.add_completed_sub_task(ctx.data["act_plan_id"], ev.sub_task)
        return SubTaskResultEvent(sub_task=ev.sub_task, result=result)

    @step()
    async def gather_results(
        self, ctx: Context, ev: SubTaskResultEvent
    ) -> ExecutePlanEvent | StopEvent:
        # wait for all sub tasks to finish
        num_sub_tasks = ctx.data["num_sub_tasks"]
        results = ctx.collect_events(ev, [SubTaskResultEvent] * num_sub_tasks)
        if results is None:
            return None

        # store all results for refining the plan
        ctx.data["results"] = ctx.data.get("results", {})
        for result in results:
            ctx.data["results"][result.sub_task.name] = result.result

        upcoming_sub_tasks = self.planner.state.get_next_sub_tasks(
            ctx.data["act_plan_id"]
        )
        # if no more tasks to do, stop workflow and send result of last step
        if len(upcoming_sub_tasks) == 0:
            return StopEvent(result=results[-1].result)

        if self.refine_plan:
            new_plan = await self.planner.refine_plan(
                ctx.data["task"], ctx.data["act_plan_id"], ctx.data["results"]
            )
            # inform about the new plan
            if new_plan is not None:
                ctx.session.write_event_to_stream(
                    PlanEvent(event_type=PlanEventType.REFINED, plan=new_plan)
                )

        # continue executing plan
        return ExecutePlanEvent()
