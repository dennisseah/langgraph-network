import asyncio
import logging

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langgraph_network.hosting import container
from langgraph_network.tools.agents import (
    customer_agent,
    investment_account_agent,
    saving_account_agent,
)

CUSTOMER_AGENT = "customer_agent"
INVESTMENT_ACCOUNT_ASSISTANT = "investment_account_assistant"
SAVING_ACCOUNT_ASSISTANT = "saving_account_assistant"


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


class Graph:
    def __init__(self):
        self.graph = self.create_graph()
        self.logger = container[logging.Logger]

    def create_node(self, node_name: str, agent: CompiledGraph, next_node_name: str):
        """
        Create a node for the graph.

        :param node_name: The name of the node.
        :param agent: The agent to invoke.
        :param next_node_name: The name of the next node.
        """

        def wrapper(
            state: MessagesState,
        ) -> Command:  # type: ignore
            result = agent.invoke(state)
            goto = get_next_node(result["messages"][-1], next_node_name)
            result["messages"][-1] = HumanMessage(
                content=result["messages"][-1].content, name=node_name
            )
            return Command(
                update={"messages": result["messages"]},
                goto=goto,
            )

        return wrapper

    def create_graph(self) -> CompiledStateGraph:
        """
        Create the graph for the workflow.

        Create the nodes and edge for the workflow.
        """
        workflow = StateGraph(MessagesState)

        workflow.add_node(
            CUSTOMER_AGENT,
            self.create_node(
                CUSTOMER_AGENT, customer_agent, INVESTMENT_ACCOUNT_ASSISTANT
            ),
        )
        workflow.add_node(
            INVESTMENT_ACCOUNT_ASSISTANT,
            self.create_node(
                INVESTMENT_ACCOUNT_ASSISTANT,
                investment_account_agent,
                SAVING_ACCOUNT_ASSISTANT,
            ),
        )
        workflow.add_node(
            SAVING_ACCOUNT_ASSISTANT,
            self.create_node(
                SAVING_ACCOUNT_ASSISTANT, saving_account_agent, CUSTOMER_AGENT
            ),
        )

        # above a create 3 nodes and here we create the edge
        workflow.add_edge(START, CUSTOMER_AGENT)
        return workflow.compile()

    async def astream(self):
        async for state in self.graph.astream(
            {
                "messages": [
                    HumanMessage(
                        content="Get my investment and saving account balance. "
                        "And sum them up.",
                    )
                ]
            },
            {"recursion_limit": 10},
        ):
            self.logger.info(f"[STREAMING]: {next(iter(state.keys()))}")
            for message in state.values():
                for msg in message.values():
                    for m in msg:
                        if m.content:
                            self.logger.info(f"[STREAMING]:     {m.type}: {m.content}")


if __name__ == "__main__":
    graph = Graph()
    asyncio.run(graph.astream())
