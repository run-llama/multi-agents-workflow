from typing import List, Union
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
)


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
