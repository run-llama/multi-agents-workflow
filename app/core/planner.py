from typing import Dict, List, Union
import uuid


from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import BaseTool
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.bridge.pydantic import ValidationError
from llama_index.core.agent.runner.planner import (
    SubTask,
    Plan,
    PlannerAgentState,
    DEFAULT_INITIAL_PLAN_PROMPT,
    DEFAULT_PLAN_REFINE_PROMPT,
)


class Planner:
    def __init__(
        self,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        initial_plan_prompt: Union[str, PromptTemplate] = DEFAULT_INITIAL_PLAN_PROMPT,
        plan_refine_prompt: Union[str, PromptTemplate] = DEFAULT_PLAN_REFINE_PROMPT,
        verbose: bool = True,
    ) -> None:
        if llm is None:
            llm = Settings.llm
            # Uncomment to fix tools_by_name KeyError
            # from llama_index.llms.openai import OpenAI
            # llm = OpenAI()
        self.llm = llm
        assert self.llm.metadata.is_function_calling_model

        self.tools = tools or []
        self.state = PlannerAgentState()
        self.verbose = verbose

        if isinstance(initial_plan_prompt, str):
            initial_plan_prompt = PromptTemplate(initial_plan_prompt)
        self.initial_plan_prompt = initial_plan_prompt

        if isinstance(plan_refine_prompt, str):
            plan_refine_prompt = PromptTemplate(plan_refine_prompt)
        self.plan_refine_prompt = plan_refine_prompt

    async def create_plan(self, input: str) -> str:
        tools = self.tools
        tools_str = ""
        for tool in tools:
            tools_str += tool.metadata.name + ": " + tool.metadata.description + "\n"

        try:
            plan = await self.llm.astructured_predict(
                Plan,
                self.initial_plan_prompt,
                tools_str=tools_str,
                task=input,
            )
        except (ValueError, ValidationError):
            if self.verbose:
                print("No complex plan predicted. Defaulting to a single task plan.")
            plan = Plan(
                sub_tasks=[
                    SubTask(
                        name="default", input=input, expected_output="", dependencies=[]
                    )
                ]
            )

        if self.verbose:
            print("=== Initial plan ===")
            for sub_task in plan.sub_tasks:
                print(
                    f"{sub_task.name}:\n{sub_task.input} -> {sub_task.expected_output}\ndeps: {sub_task.dependencies}\n\n"
                )

        plan_id = str(uuid.uuid4())
        self.state.plan_dict[plan_id] = plan

        return plan_id

    async def refine_plan(
        self,
        input: str,
        plan_id: str,
        completed_sub_tasks: Dict[str, str],
    ) -> None:
        """Refine a plan."""
        prompt_args = self.get_refine_plan_prompt_kwargs(
            plan_id, input, completed_sub_tasks
        )

        # Uncomment to fix tools_by_name KeyError
        # print(self.plan_refine_prompt.format(**prompt_args))

        try:
            new_plan = await self.llm.astructured_predict(
                Plan, self.plan_refine_prompt, **prompt_args
            )

            self._update_plan(plan_id, new_plan)
        except (ValueError, ValidationError) as e:
            # likely no new plan predicted
            if self.verbose:
                print(f"No new plan predicted: {e}")
            return

    def _update_plan(self, plan_id: str, new_plan: Plan) -> None:
        """Update the plan."""
        # update state with new plan
        self.state.plan_dict[plan_id] = new_plan

        if self.verbose:
            print("=== Refined plan ===")
            for sub_task in new_plan.sub_tasks:
                print(
                    f"{sub_task.name}:\n{sub_task.input} -> {sub_task.expected_output}\ndeps: {sub_task.dependencies}\n\n"
                )

    def get_refine_plan_prompt_kwargs(
        self,
        plan_id: str,
        task: str,
        completed_sub_task: Dict[str, str],
    ) -> dict:
        """Get the refine plan prompt."""
        # gather completed sub-tasks and response pairs
        completed_outputs_str = ""
        for sub_task_name, task_output in completed_sub_task.items():
            task_str = f"{sub_task_name}:\n" f"\t{task_output!s}\n"
            completed_outputs_str += task_str

        # get a string for the remaining sub-tasks
        remaining_sub_tasks = self.state.get_remaining_subtasks(plan_id)
        remaining_sub_tasks_str = "" if len(remaining_sub_tasks) != 0 else "None"
        for sub_task in remaining_sub_tasks:
            task_str = (
                f"SubTask(name='{sub_task.name}', "
                f"input='{sub_task.input}', "
                f"expected_output='{sub_task.expected_output}', "
                f"dependencies='{sub_task.dependencies}')\n"
            )
            remaining_sub_tasks_str += task_str

        # get the tools string
        tools = self.tools
        tools_str = ""
        for tool in tools:
            tools_str += tool.metadata.name + ": " + tool.metadata.description + "\n"

        # return the kwargs
        return {
            "tools_str": tools_str.strip(),
            "task": task.strip(),
            "completed_outputs": completed_outputs_str.strip(),
            "remaining_sub_tasks": remaining_sub_tasks_str.strip(),
        }
