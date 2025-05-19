from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain.tools import tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

from langchain.schema import PromptValue, LLMResult, Generation
from typing import Optional, List

import re
import json

import sys
sys.path.append('../')
from clustering_agent import ClusteringAgent

class SimpleLLM(BaseLanguageModel):  
    def predict(self, prompt: PromptValue, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt.to_string(), stop=stop)

    def invoke(self, inputs: dict) -> str:
        # "input" key with prompt string
        prompt_str = inputs.get("input", "")
        return self._call(prompt_str)

    async def apredict(self, prompt: PromptValue, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt.to_string(), stop=stop)

    def generate_prompt(self, prompt: PromptValue, stop: Optional[List[str]] = None) -> LLMResult:
        text = self._call(prompt.to_string(), stop=stop)
        generation = Generation(text=text)
        return LLMResult(generations=[[generation]])

    async def agenerate_prompt(self, prompt: PromptValue, stop: Optional[List[str]] = None) -> LLMResult:
        return self.generate_prompt(prompt, stop=stop)

    def predict_messages(self, prompt: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> BaseMessage:
        # Last message in the conversation
        last_msg = prompt[-1]
    
        # Last message from the tool (FunctionMessage)? don't trigger tool call again (avoid loops)
        if isinstance(last_msg, FunctionMessage):
            return AIMessage(content="\nThanks!")
    
        # From the user, and has words CLUSTERING and a CSV file, trigger the tool.
        if isinstance(last_msg, HumanMessage):
            content = last_msg.content.lower()
            files_csv = re.findall(r'\b[\w\-\.]+\.csv\b', content)

            #This is a simple parser, a simple rule-based prediction
            if ("clusteriza" in content or "clustering" in content) and files_csv:
                return AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "clusterize_dataset",
                            "arguments": json.dumps({"path_csv": files_csv[0]})
                        }
                    }
                )

        return AIMessage(content="Sorry, I don't understand the instruction.")

    async def apredict_messages(self, prompt: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> BaseMessage:
        return self.predict_messages(prompt, stop=stop)

    @property
    def _llm_type(self) -> str:
        return "simple-llm"


# Define a LangChain tool that uses the ClusteringAgent
@tool("clusterize_dataset")
def clusterize_dataset(path_csv: str) -> str:
    """Load the CSV file and run the clustering pipeline."""
    try:
    	agent = ClusteringAgent(path_csv)
    	agent.run_pipeline(show_figure=False, save_figure=True)
    	agent.save_results('output.csv')  #TODO: move it to an user argument.
    	msg = f"Clustering completed: {agent.best_method} with {agent.n_clusters} clusters."
    	msg += f"Output saved in ./output.csv"
    	return msg
    except Exception as e:
        print(f"Error when running ClusteringAgent: {str(e)}")

if __name__ == '__main__':
    # Initialize agent with the SimpleLLM model and tthe LC tool
    llm = SimpleLLM()
    tools = [clusterize_dataset]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    # Run it
    print('\nRunning Agent')
    user_message = "¿Podrias clusterizar el archivo iris_data_challenge.csv?"
    response = agent.invoke(user_message)

    #print(response)
