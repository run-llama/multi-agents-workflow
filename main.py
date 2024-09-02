# flake8: noqa: E402
import asyncio
import textwrap
from dotenv import load_dotenv
from app.agents.single import AgentRunResult
from app.examples.choreography import create_choreography
from app.settings import init_settings

from app.examples.orchestrator import create_orchestrator

load_dotenv()
init_settings()


def info(prefix: str, text: str) -> None:
    truncated = textwrap.shorten(text, width=255, placeholder="...")
    print(f"[{prefix}] {truncated}")


async def main():
    agent = create_choreography()
    # agent = create_orchestrator()
    task = asyncio.create_task(
        agent.run(input="Write a blog post about physical standards for letters")
    )

    async for ev in agent.stream_events():
        info(ev.name, ev.msg)

    ret: AgentRunResult = await task
    print(f"\n\nResult:\n\n{ret.response.message.content}")


if __name__ == "__main__":
    asyncio.run(main())
