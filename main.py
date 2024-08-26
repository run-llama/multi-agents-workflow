# flake8: noqa: E402
import asyncio
from dotenv import load_dotenv
from app.agents.researcher.agent import get_query_engine_tool
from app.core.agent_call import AgentCallingAgent
from app.core.function_call import FunctionCallingAgent
from app.settings import init_settings


load_dotenv()
init_settings()


async def main():
    researcher = FunctionCallingAgent(
        name="researcher",
        tools=[get_query_engine_tool()],
        system_prompt="You are a researcher agent. You are given a researching task. You must use your tools to complete the research.",
    )
    reviewer = AgentCallingAgent(
        name="reviewer",
        system_prompt="You are an expert in reviewing blog posts. You are given a task to write a blog post. Before starting to write the post, consult the researcher agent to get the information you need. Don't make up any information yourself.",
    )
    writer = AgentCallingAgent(
        name="writer",
        agents=[researcher, reviewer],
        system_prompt="""You are an expert in writing blog posts. You are given a task to write a blog post. Before starting to write the post, consult the researcher agent to get the information you need. Don't make up any information yourself.
        After creating a draft for the post, send it to the reviewer agent to receive some feedback and make sure to incorporate the feedback from the reviewer.
        You can consult the reviewer and researcher multiple times. Only finish the task once the reviewer is satisfied.""",
    )
    ret = await writer.run(input="Write a blog post about letter standards")
    print(ret["response"])


if __name__ == "__main__":
    asyncio.run(main())
