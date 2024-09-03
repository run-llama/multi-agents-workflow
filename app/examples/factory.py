from app.examples.choreography import create_choreography
from app.examples.orchestrator import create_orchestrator
from app.examples.workflow import create_workflow


from llama_index.core.workflow import Workflow


import os


def create_agent() -> Workflow:
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

    return agent
