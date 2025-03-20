from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOpenAI
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

# Tool definitions with proper docstrings
@tool
def get_current_datetime():
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def CalculatorTool(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Calculator supports operations +, -, *, /, **
    
    Args:
        expression: A mathematical expression as a string
        
    Returns:
        The result of the calculation
    """
    calculator = Calculator()
    return str(calculator.evaluate_expression(expression))

@tool
def WeatherTool(place: str) -> str:
    """Get the current weather for a specific location.
    
    Args:
        place: Name of city, country, or location
        
    Returns:
        Weather information for the specified location
    """
    tool = TavilySearchResults(max_results=1)
    return tool.invoke({'query': f'what is the current weather in {place}'})

# State definition
class WorkflowState(MessagesState):
    """State for our workflow."""
    preprocessing_result: str = ""
    final_result: str = ""

class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents."""
    
    def __init__(self):
        """Initialize the ToolUsingAgentChat with empty attributes."""
        self.llm = None
        self.agent_executor = None
        self.tools = []
    
    def initialize(self) -> None:
        """Initialize components for the tool-using agent.
        
        Sets up the language model, tools, and creates the agent.
        """
        # Initialize the language model
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Create the agent with tools
        self._create_agent()
    
    def _create_tools(self) -> List[Any]:
        """Create and return the list of tools for the agent.
        
        Returns:
            List of tool functions ready for the agent to use
        """
        # Return the pre-defined tool functions
        return [CalculatorTool, get_current_datetime, WeatherTool]
    
    def _create_agent(self) -> None:
        """Create and set up the ReAct agent executor.
        
        Uses the LangChain Hub's React prompt template for consistency.
        """
        # Pull the standard React prompt from LangChain Hub
        prompt = hub.pull("hwchase17/react")
        
        # Create the React agent with our tools and LLM
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Set up the agent executor with appropriate parameters
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a user message using the tool-using agent.
        
        Args:
            message: The user's input message
            chat_history: Optional list of previous chat messages
            
        Returns:
            str: The assistant's response after using appropriate tools
        """
        # Prepare input context
        input_context = {"input": message}
        
        # Include chat history if available
        if chat_history:
            input_context["chat_history"] = chat_history
            
        # Run the agent and get the response
        response = self.agent_executor.invoke(input_context)
        
        # Return the output or fallback to string representation
        return response.get("output", str(response))