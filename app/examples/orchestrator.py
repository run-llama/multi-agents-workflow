from app.agents.single import FunctionCallingAgent
from app.agents.multi import AgentOrchestrator
from app.examples.researcher import create_researcher


def create_orchestrator():
    researcher = create_researcher()
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
