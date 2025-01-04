"""Defines our top level DI container.
Utilizes the Lagom library for dependency injection, see more at:

- https://lagom-di.readthedocs.io/en/latest/
- https://github.com/meadsteve/lagom
"""

import logging
import os

from dotenv import load_dotenv
from lagom import Container, dependency_definition

from langgraph_network.protocols.i_azure_openai_service import IAzureOpenAIService

load_dotenv(dotenv_path=".env")


container = Container()
"""The top level DI container for our application."""


# Register our dependencies ------------------------------------------------------------


@dependency_definition(container, singleton=True)
def logger() -> logging.Logger:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    logging.Formatter(fmt=" %(name)s :: %(levelname)-8s :: %(message)s")
    return logging.getLogger("langgraph_network")


@dependency_definition(container, singleton=True)
def azure_openai_service() -> IAzureOpenAIService:
    from langgraph_network.services.azure_openai_service import (
        AzureOpenAIService,
    )

    return container[AzureOpenAIService]
