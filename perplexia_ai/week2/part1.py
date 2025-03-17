from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List, Dict

from perplexia_ai.core.chat_interface import ChatInterface

tavily_search = TavilySearchResults(max_results=5,include_answer=True,include_raw_content=True )

class GraphState(TypedDict):
    question: str
    search_results: TavilySearchResults
    answer: str

def WebSearchResults(state: GraphState):
    question= state['question']
    search_results=tavily_search.invoke(question)
    return {"search_results": search_results} 

def SummarizeWebSearchResults(state: GraphState):
    search_results = state['search_results']
    system_prompt= """
    You are an assitant to summarize the provided {search_results}. 
    For each item in the collection extract title of the article , 
    Brief summary of the contant in less than 50 words,
    and url"""
    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    answer = llm.invoke(prompt_template.format(search_results=search_results))
    return {"answer": answer}  
    
class UseWebSearch(ChatInterface):
    def __init__(self):
        super().__init__()
        self.graph = None
    
    def initialize(self):
        graph_builder = StateGraph(GraphState)
        graph_builder.add_node("webSerch" , WebSearchResults)
        graph_builder.add_node("summarize" , SummarizeWebSearchResults)

        graph_builder.add_edge(START, "webSerch")
        graph_builder.add_edge("webSerch", "summarize")
        graph_builder.add_edge("summarize", END)
        self.graph = graph_builder.compile()
    
    
    def process_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> str:
        
        events = self.graph.stream(
            {"question": message},
            stream_mode="values",
        )
        response = ""
        for event in events:
            if "answer" in event:  # Changed to look for "generation"
                response = event["answer"].content  # Changed to extract from "generation"
                break
        return response