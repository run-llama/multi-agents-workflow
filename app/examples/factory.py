import logging
from app.examples.choreography import create_choreography
from app.examples.orchestrator import create_orchestrator
from app.examples.workflow import create_workflow


from llama_index.core.workflow import Workflow


import os

logger = logging.getLogger("uvicorn")


def create_agent() -> Workflow:
    agent_type = os.getenv("EXAMPLE_TYPE", "").lower()
    match agent_type:
        case "choreography":
            agent = create_choreography()
        case "orchestrator":
            agent = create_orchestrator()
        case "workflow":
            agent = create_workflow()
        case _:
            raise ValueError(
                f"Invalid EXAMPLE_TYPE env variable: {agent_type}. Choose 'choreography', 'orchestrator', or 'workflow'."
            )

    logger.info(f"Using agent pattern: {agent_type}")

    return agent
