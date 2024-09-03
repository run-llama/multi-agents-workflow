import asyncio
import logging
import os
import textwrap
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from llama_index.core.chat_engine.types import BaseChatEngine, NodeWithScore
from llama_index.core.llms import MessageRole
from llama_index.core.llms import ChatResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from app.agents.single import AgentRunResult
from app.api.routers.events import EventCallbackHandler
from app.api.routers.models import (
    ChatData,
    Message,
    Result,
    SourceNodes,
)
from app.api.routers.vercel_response import VercelStreamResponse
from app.examples.choreography import create_choreography
from app.examples.orchestrator import create_orchestrator
from app.examples.workflow import create_workflow

chat_router = r = APIRouter()

logger = logging.getLogger("uvicorn")


def info(prefix: str, text: str) -> None:
    truncated = textwrap.shorten(text, width=255, placeholder="...")
    print(f"[{prefix}] {truncated}")


async def run_workflow(input: str) -> ChatResponse:
    TYPE = os.getenv("EXAMPLE_TYPE", "").lower()
    match TYPE:
        case "choreography":
            agent = create_choreography()
        case "orchestrator":
            agent = create_orchestrator()
        case "workflow":
            agent = create_workflow()
        case _:
            raise ValueError(
                f"Invalid EXAMPLE_TYPE env variable: {TYPE}. Choose 'choreography', 'orchestrator', or 'workflow'."
            )

    task = asyncio.create_task(agent.run(input=input))

    async for ev in agent.stream_events():
        info(ev.name, ev.msg)

    ret: AgentRunResult = await task
    return ret.response


# streaming endpoint - delete if not needed
@r.post("")
async def chat(
    request: Request,
    data: ChatData,
):
    try:
        last_message_content = data.get_last_message_content()
        # TODO: use message history
        # messages = data.get_history_messages()
        # TODO: generate filters based on doc_ids
        # for now just use all documents
        # doc_ids = data.get_chat_document_ids()
        # TODO: use params
        # params = data.data or {}

        event_handler = EventCallbackHandler()
        response = await run_workflow(last_message_content)

        # TODO: Hack, convert chat response to streaming chat response
        stream_response = StreamingAgentChatResponse()
        stream_response._ensure_async_setup()
        for tok in response.message.content:
            stream_response.aput_in_queue(tok)
        stream_response.is_done = True

        return VercelStreamResponse(request, event_handler, stream_response, data)
    except Exception as e:
        logger.exception("Error in chat engine", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat engine: {e}",
        ) from e
