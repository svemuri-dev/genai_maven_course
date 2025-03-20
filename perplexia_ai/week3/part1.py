from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langgraph.graph import MessagesState
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain import hub
@tool
def get_current_datetime():
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@tool
def CalculatorTool(expression: str) -> float:
    """Evaluate a valid mathematical expression in Python and return the result.Calculator only supports operations +, -, *, /, **"""
    calculator = Calculator()
    return str(calculator.evaluate_expression(expression))
@tool
def WeatherTool(place: str) -> float:
    """Evaluate a valid mathematical expression in Python and return the result.Calculator only supports operations +, -, *, /, **"""
    tool = TavilySearchResults(
            max_results=1
        )
    return tool.invoke({'query': 'what is weather in ' + place})

# Create our custom state
class WorkflowState(MessagesState):
    """State for our workflow."""
    preprocessing_result: str = ""
    final_result: str = ""

class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents."""
    
    def __init__(self):
        self.llm = None
        self.agent_executor = None
        self.tools = []
    
    def initialize(self) -> None:
        """Initialize components for the tool-using agent."""
        # Initialize chat model with updated import
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create the ReAct agent
        self._create_agent()
    
    def _create_tools(self) -> List[Any]:
        """Create and return the list of tools for the agent."""
        # Calculator tool
       

        # DateTime tool
   

        # Weather tool using Tavily


        tools = [CalculatorTool, get_current_datetime, WeatherTool]
        return tools
    
    def _create_agent(self) -> None:
        """Create and set up the ReAct agent executor."""
        
        # Use the default ReAct prompt instead of creating a custom one
        # This will automatically include the required variables
        
       
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Set up the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the tool-using agent."""
        # Prepare input context from chat history if available
        input_context = {"input": message}  # Changed from "question" to "input"
        if chat_history:
            input_context["chat_history"] = chat_history
            
        # Run the agent and get the result
        response = self.agent_executor.invoke(input_context)
        
        # Extract the final response text
        if "output" in response:
            return response["output"]
        else:
            # Fallback
            return str(response)