from PropBots.logger import logging
from PropBots.CustomPropException import PropBotException
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain.prompts import PromptTemplate
import time

import warnings
warnings.filterwarnings('ignore')


def AgentExecuter(data, llm, max_value_length, max_iterations_limit):
    """
    Initializes and returns an agent to process JSON-based queries.

    Args:
        data (dict): The data to be processed by the agent.
        llm: The language model used to process queries.
        max_value_length (int): Maximum allowed length of values in the JSON.
        max_iterations_limit (int): Maximum number of iterations the agent can perform.

    Returns:
        agent: The initialized JSON agent.
    """
    try:
        spec = JsonSpec(dict_=data, max_value_length=max_value_length)
        toolkit = JsonToolkit(spec=spec)
        agent = create_json_agent(llm=llm, toolkit=toolkit, max_iterations=max_iterations_limit, verbose=False)
        logging.info("Agent created successfully.")
        return agent

    except Exception as e:
        logging.error(f"Error creating agent: {e}")
        raise PropBotException("Failed to create JSON agent") from e


# Prompt template for the property expert agent
temp2 = """You're a property expert. Answer the given query. \n##Query: {query}"""


def ProcessQuery(query, agent, retries=3, delay=10):
    """
    Processes a given query using the provided agent with retry logic.

    Args:
        query (str): The query to be processed.
        agent: The agent responsible for handling the query.
        retries (int): Number of retry attempts in case of failure (default is 3).
        delay (int): Delay (in seconds) between retries (default is 10 seconds).

    Returns:
        str: The result of the query processing or an error message in case of failure.
    """
    attempt = 0
    while attempt < retries:
        try:
            # Format the query using the prompt template
            formatted_query = PromptTemplate(input_variables=["query"], template=temp2).format(query=query)

            # Invoke the agent with the formatted query
            logging.info(f"Processing query: '{query}' (Attempt {attempt + 1}/{retries})")
            result = agent.invoke(formatted_query)
            logging.info(f"Query processed successfully: '{query}'")
            return result['output']

        except Exception as e:
            attempt += 1
            logging.error(f"Attempt {attempt} failed for query '{query}': {str(e)}")
            print(f"Attempt {attempt} failed for query '{query}': {str(e)}")
            if attempt < retries:
                time.sleep(delay)  # Wait before retrying
                logging.info(f"Retrying query '{query}' after {delay} seconds...")
            else:
                logging.error(f"All {retries} attempts failed for query: '{query}'")
                return f"Error: Unable to process query after {retries} attempts. Please check error_log.txt for details."

