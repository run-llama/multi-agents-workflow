# flake8: noqa: E402
import asyncio
import os
import textwrap
from typing import AsyncGenerator
from dotenv import load_dotenv

from app.config import DATA_DIR
from app.examples.factory import create_agent

load_dotenv()

import logging

import uvicorn
from app.api.routers.chat import chat_router
from app.api.routers.chat_config import config_router
from app.api.routers.upload import file_upload_router
from app.observability import init_observability
from app.settings import init_settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

init_settings()
init_observability()


environment = os.getenv("ENVIRONMENT", "dev")  # Default to 'development' if not set
logger = logging.getLogger("uvicorn")

if environment == "dev":
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Redirect to documentation page when accessing base URL
    @app.get("/")
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")


def mount_static_files(directory, path):
    if os.path.exists(directory):
        logger.info(f"Mounting static files '{directory}' at '{path}'")
        app.mount(
            path,
            StaticFiles(directory=directory, check_dir=False),
            name=f"{directory}-static",
        )


# Mount the data files to serve the file viewer
mount_static_files(DATA_DIR, "/api/files/data")
# Mount the output files from tools
mount_static_files("output", "/api/files/output")

app.include_router(chat_router, prefix="/api/chat")
app.include_router(config_router, prefix="/api/chat/config")
app.include_router(file_upload_router, prefix="/api/chat/upload")


def run_api():
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", "8000"))
    reload = True if environment == "dev" else False

    uvicorn.run(app="main:app", host=app_host, port=app_port, reload=reload)


async def main():
    def info(prefix: str, text: str) -> None:
        truncated = textwrap.shorten(text, width=255, placeholder="...")
        print(f"[{prefix}] {truncated}")

    agent = create_agent()

    task = asyncio.create_task(
        agent.run(
            input="Write a blog post about physical standards for letters",
            streaming=True,
        )
    )

    async for ev in agent.stream_events():
        info(ev.name, ev.msg)

    ret: AsyncGenerator = await task
    async for token in ret:
        print(token.delta, end="", flush=True)

    # ret: AgentRunResult = await task
    # print(ret.response.message.content)


if __name__ == "__main__":
    if os.getenv("FAST_API", "false").lower() == "false":
        asyncio.run(main())
    else:
        run_api()
