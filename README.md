This is a [LlamaIndex](https://www.llamaindex.ai/) multi-agents project using [Workflows](https://docs.llamaindex.ai/en/stable/understanding/workflows/).

## Overview

This example is using three agents to generate a blog post:

- a researcher that retrieves content via a RAG pipeline,
- a writer that specializes in writing blog posts and
- a reviewer that is reviewing the blog post.

There are three different methods how the agents can interact to reach their goal:

1. [Choreography](./app/examples/choreography.py) - the agents decide themselves to delegate a task to another agent
1. [Orchestator](./app/examples/orchestrator.py) - a central orchestrator decides which agent should execute a task
1. [Explicit Workflow](./app/examples/workflow.py) - a pre-defined workflow specific for the task is used to execute the tasks

## Getting Started

First, setup the environment with poetry:

> **_Note:_** This step is not needed if you are using the dev-container.

```shell
poetry install
```

Then check the parameters that have been pre-configured in the `.env` file in this directory. (E.g. you might need to configure an `OPENAI_API_KEY` if you're using OpenAI as model provider).

Second, generate the embeddings of the documents in the `./data` directory:

```shell
poetry run generate
```

Third, run the agents in one command:

```shell
poetry run python main.py
```

Per default, the example is using the explicit workflow. You can change the example by setting the `EXAMPLE_TYPE` environment variable to `choreography` or `orchestrator`.

To add an API endpoint, set the `FAST_API` environment variable to `true`.

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex.
- [Workflows Introduction](https://docs.llamaindex.ai/en/stable/understanding/workflows/) - learn about LlamaIndex workflows.

You can check out [the LlamaIndex GitHub repository](https://github.com/run-llama/llama_index) - your feedback and contributions are welcome!
