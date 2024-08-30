# flake8: noqa: E402
import asyncio
import os
from dotenv import load_dotenv
from app.core.agent_call import AgentCallingAgent, AgentOrchestrator
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


def create_choreography():
    researcher = FunctionCallingAgent(
        name="researcher",
        tools=[get_query_engine_tool()],
        role="expert in retrieving any unknown content",
        system_prompt="You are a researcher agent. You are given a researching task. You must use your tools to complete the research.",
    )
    reviewer = FunctionCallingAgent(
        name="reviewer",
        role="expert in reviewing blog posts",
        system_prompt="You are an expert in reviewing blog posts. You are given a task to review a blog post. Review the post for logical inconsistencies, ask critical questions, and provide suggestions for improvement. Furthermore, proofread the post for grammar and spelling errors. If the post is good, you can say 'The post is good.'",
    )
    return AgentCallingAgent(
        name="writer",
        agents=[researcher, reviewer],
        role="expert in writing blog posts",
        system_prompt="""You are an expert in writing blog posts. You are given a task to write a blog post. Before starting to write the post, consult the researcher agent to get the information you need. Don't make up any information yourself.
        After creating a draft for the post, send it to the reviewer agent to receive some feedback and make sure to incorporate the feedback from the reviewer.
        You can consult the reviewer and researcher multiple times. Only finish the task once the reviewer is satisfied.""",
    )


def create_orchestrator():
    researcher = FunctionCallingAgent(
        name="researcher",
        tools=[get_query_engine_tool()],
        role="expert in retrieving any unknown content",
        system_prompt="You are a researcher agent. You are given a researching task. You must use your tools to complete the research.",
    )
    writer = FunctionCallingAgent(
        name="writer",
        role="expert in writing blog posts",
        system_prompt="""You are an expert in writing blog posts. You are given a task to write a blog post. Don't make up any information yourself. If you don't have the necessary information to write a blog post, reply "I need information about the topic to write the blog post". If you have all the information needed, write the blog post.""",
    )
    reviewer = FunctionCallingAgent(
        name="reviewer",
        role="expert in reviewing blog posts",
        system_prompt="""You are an expert in reviewing blog posts. You are given a task to review a blog post. Review the post and fix the issues found yourself. You must output a final blog post.
        Especially check for logical inconsistencies and proofread the post for grammar and spelling errors.""",
    )
    return AgentOrchestrator(
        agents=[writer, reviewer, researcher],
        refine_plan=False,
    )


async def main():
    # agent = create_choreography()
    agent = create_orchestrator()
    ret = await agent.run(
        input="Write a blog post about physical standards for letters"
    )
    print(ret)


if __name__ == "__main__":
    asyncio.run(main())
