from langgraph.prebuilt import create_react_agent

from langgraph_network.hosting import container
from langgraph_network.protocols.i_azure_openai_service import (
    IAzureOpenAIService,
)
from langgraph_network.tools.system_prompt import make_system_prompt
from langgraph_network.tools.tools import (
    get_bank_account_id,
    get_investment_account_balance,
    get_saving_account_balance,
)

llm_model = container[IAzureOpenAIService].get_model()

customer_agent = create_react_agent(
    llm_model,
    tools=[
        get_bank_account_id,
    ],
    state_modifier=make_system_prompt("You can provide bank account ID."),
)

investment_account_agent = create_react_agent(
    llm_model,
    [get_investment_account_balance],
    state_modifier=make_system_prompt("You can provide investment account balance."),
)

saving_account_agent = create_react_agent(
    llm_model,
    [get_saving_account_balance],
    state_modifier=make_system_prompt("You can provide saving account balance."),
)
