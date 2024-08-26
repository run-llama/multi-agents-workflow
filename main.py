# flake8: noqa: E402
import asyncio
import os
from dotenv import load_dotenv
from app.core.agent_call import AgentCallingAgent
from app.core.function_call import FunctionCallingAgent
from app.engine.index import get_index
from app.settings import init_settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata

load_dotenv()
init_settings()


def get_query_engine_tool() -> QueryEngineTool:
    """
    Provide an agent worker that can be used to query the index.
    """
    index = get_index()
    if index is None:
        raise ValueError("Index not found. Please create an index first.")
    top_k = int(os.getenv("TOP_K", 0))
    query_engine = index.as_query_engine(
        **({"similarity_top_k": top_k} if top_k != 0 else {})
    )
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_index",
            description="""
                Use this tool to retrieve information about the text corpus from the index.
            """,
        ),
    )


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
        verbose=False,
        system_prompt="""You are an expert in writing blog posts. You are given a task to write a blog post. Before starting to write the post, consult the researcher agent to get the information you need. Don't make up any information yourself.
        After creating a draft for the post, send it to the reviewer agent to receive some feedback and make sure to incorporate the feedback from the reviewer.
        You can consult the reviewer and researcher multiple times. Only finish the task once the reviewer is satisfied.""",
    )
    ret = await writer.run(input="Write a blog post about letter standards")
    print(ret["response"])


if __name__ == "__main__":
    asyncio.run(main())
