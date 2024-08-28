from typing import Dict, List, Union
import uuid


from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import BaseTool
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.bridge.pydantic import ValidationError
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """A single sub-task in a plan."""

    name: str = Field(..., description="The name of the sub-task.")
    agent_name: str | None = Field(
        ..., description="The name of the agent to use for the sub-task."
    )
    input: str = Field(
        ..., description="The input to give to the agent to solve the sub-task."
    )
    dependencies: List[str] = Field(
        ...,
        description="The sub-task names that must be completed before this sub-task.",
    )


class Plan(BaseModel):
    """A series of sub-tasks delegated to other agents to accomplish an overall task."""

    sub_tasks: List[SubTask] = Field(
        ..., description="The sub-tasks in the plan that are delegated to agents."
    )


class PlannerAgentState(BaseModel):
    """Agent state."""

    plan_dict: Dict[str, Plan] = Field(
        default_factory=dict, description="An id-plan lookup."
    )
    completed_sub_tasks: Dict[str, List[SubTask]] = Field(
        default_factory=dict, description="A list of completed sub-tasks for each plan."
    )

    def get_completed_sub_tasks(self, plan_id: str) -> List[SubTask]:
        return self.completed_sub_tasks.get(plan_id, [])

    def add_completed_sub_task(self, plan_id: str, sub_task: SubTask) -> None:
        if plan_id not in self.completed_sub_tasks:
            self.completed_sub_tasks[plan_id] = []

        self.completed_sub_tasks[plan_id].append(sub_task)

    def get_next_sub_tasks(self, plan_id: str) -> List[SubTask]:
        next_sub_tasks: List[SubTask] = []
        plan = self.plan_dict[plan_id]

        if plan_id not in self.completed_sub_tasks:
            self.completed_sub_tasks[plan_id] = []

        completed_sub_tasks = self.completed_sub_tasks[plan_id]
        completed_sub_task_names = [sub_task.name for sub_task in completed_sub_tasks]

        for sub_task in plan.sub_tasks:
            dependencies_met = all(
                dep in completed_sub_task_names for dep in sub_task.dependencies
            )

            if sub_task.name not in completed_sub_task_names and dependencies_met:
                next_sub_tasks.append(sub_task)
        return next_sub_tasks

    def get_remaining_subtasks(self, plan_id: str) -> List[SubTask]:
        remaining_subtasks = []
        plan = self.plan_dict[plan_id]

        if plan_id not in self.completed_sub_tasks:
            self.completed_sub_tasks[plan_id] = []

        completed_sub_tasks = self.completed_sub_tasks[plan_id]
        completed_sub_task_names = [sub_task.name for sub_task in completed_sub_tasks]

        for sub_task in plan.sub_tasks:
            if sub_task.name not in completed_sub_task_names:
                remaining_subtasks.append(sub_task)
        return remaining_subtasks

    def reset(self) -> None:
        """Reset."""
        self.task_dict = {}
        self.completed_sub_tasks = {}
        self.plan_dict = {}


DEFAULT_INITIAL_PLAN_PROMPT = """\
Think step-by-step. Given a task and a set of agents, create a comprehensive, end-to-end plan consisting of a series of sub tasks to accomplish the task.
For each sub task include the name of the agent to call and the input to give to it.
The plan should end with an agent that can achieve the overall task.

The agents available are:
{agents_str}

Overall Task: {task}
"""


class Planner:
    def __init__(
        self,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        initial_plan_prompt: Union[str, PromptTemplate] = DEFAULT_INITIAL_PLAN_PROMPT,
        verbose: bool = True,
    ) -> None:
        if llm is None:
            llm = Settings.llm
        self.llm = llm
        assert self.llm.metadata.is_function_calling_model

        self.tools = tools or []
        self.state = PlannerAgentState()
        self.verbose = verbose

        if isinstance(initial_plan_prompt, str):
            initial_plan_prompt = PromptTemplate(initial_plan_prompt)
        self.initial_plan_prompt = initial_plan_prompt

    async def create_plan(self, input: str) -> str:
        tools = self.tools
        agents_str = ""
        for tool in tools:
            agents_str += tool.metadata.name + ": " + tool.metadata.description + "\n"

        try:
            plan = await self.llm.astructured_predict(
                Plan,
                self.initial_plan_prompt,
                agents_str=agents_str,
                task=input,
            )
            # calling astructured_predict runs into ValidationError - but apredict output does not look bad
            # print(self.initial_plan_prompt.format(agents_str=agents_str, task=input))
            # plan = await self.llm.apredict(
            #     self.initial_plan_prompt,
            #     agents_str=agents_str,
            #     task=input,
            # )
            # print(plan)
        except (ValueError, ValidationError) as e:
            print(f"Can't Generate plan: {e}")
            return None

        if self.verbose:
            print("=== Initial plan ===")
            for sub_task in plan.sub_tasks:
                print(sub_task)
                # print(
                #     f"{sub_task.name}:\n{sub_task.input} -> {sub_task.expected_output}\ndeps: {sub_task.dependencies}\n\n"
                # )

        plan_id = str(uuid.uuid4())
        self.state.plan_dict[plan_id] = plan

        print(self.state)

        return plan_id
