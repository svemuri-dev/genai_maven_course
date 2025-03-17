"""Part 3 - Conversation Memory implementation.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from perplexia_ai.tools.calculator import Calculator

class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory."""
    
      
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.full_chain = None
        self.subchains = {}
        self.calculator = Calculator()
        self.chat_history: List[Dict[str, str]] = []

    def initialize(self) -> None:
        """Initialize components for query understanding."""
        self.llm = ChatOpenAI(model_name="gpt-4", verbose=True)

        template = """You are a helpful assistant that classifies user questions.
        Here's the chat history:
        {chat_history}
        Please classify this question: "{question}"
Choose one of these types:
- "Factual" (for "What is...?", "Who invented...?" or any factual statements)
- "Analytical" (for "How does...?", "Why do...?")
- "Calculation" (for mathematical or logical calculations)
- "Comparison" (for "What's the difference between...?")
- "Definition" (for "Define...", "Explain...")

Return only the classification type as a single word."""

        self.query_classifier_prompt = PromptTemplate.from_template(template)
        classify_chain = (
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | self.query_classifier_prompt 
            | self.llm 
            | StrOutputParser()
        )

        factual_template = """Answers should be concise and direct. Put "Factual" as header 
        Here's the chat history:
        {chat_history}
        Question: {question}
Answer:"""
        analytical_template = """Analytical responses should include reasoning steps. Put "Analytical" as header
        Here's the chat history:
        {chat_history}
Question: {question}
Answer:"""
        comparison_template = """Comparisons should use structured formats (tables, bullet points). Put "Comparison" as header
        Here's the chat history:
        {chat_history}
Question: {question}
Answer:"""
        definition_template = """Definitions should include examples and use cases. Put "Definition" as header
        Here's the chat history:
        {chat_history}
Question: {question}
Answer:"""
        calculation_template = """Convert {question} to an expression that a calculator tool can evaluate and give an answer. Return only expression.
Calculator only supports operations +, -, *, /, **
        Here's the chat history:
        {chat_history}
Question: {question}
Answer:"""

        self.subchains = {
            "factual": ({ "question": lambda x: x["question"], "chat_history": lambda x: x["chat_history"]} | PromptTemplate.from_template(factual_template) | self.llm),
            "analytical": ({ "question": lambda x: x["question"], "chat_history": lambda x: x["chat_history"]} | PromptTemplate.from_template(analytical_template) | self.llm),
            "comparison": ({ "question": lambda x: x["question"], "chat_history": lambda x: x["chat_history"]} | PromptTemplate.from_template(comparison_template) | self.llm),
            "definition": ({ "question": lambda x: x["question"], "chat_history": lambda x: x["chat_history"]} | PromptTemplate.from_template(definition_template) | self.llm),
            "calculation": (
                { "question": lambda x: x["question"], "chat_history": lambda x: x["chat_history"]} | PromptTemplate.from_template(calculation_template) 
                | self.llm 
                | RunnableLambda(lambda x: x.content if isinstance(x, AIMessage) else str(x))
                | RunnableLambda(self.calculator.evaluate_expression)
            )
        }

        self.full_chain = (
            {
                "type": classify_chain, 
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            } 
            | RunnableLambda(self.route)
        )

    def route(self, info):
        """Route the question to appropriate chain based on classification."""
        try:
            chain = self.subchains.get(info["type"].lower(), self.subchains["factual"])
            response = chain.invoke({"question": info["question"], "chat_history": info["chat_history"]})
            
            # For calculation type, format the result
            if info["type"].lower() == "calculation":
                if isinstance(response, str) and response.startswith("Error"):
                    return response
                return f"The result is: {response}"
                
            # For other types, handle the response content
            if isinstance(response, (AIMessage, HumanMessage, SystemMessage)):
                return response.content
            return str(response)
        except Exception as e:
            return f"Error in processing: {str(e)}"
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding."""
        if not message:
            return "Please provide a question to classify."
            
        # Use existing chat history or initialize a new one
        if chat_history is None:
            chat_history = self.chat_history
        
        try:
            # Invoke the full chain with the message and chat history
            result = self.full_chain.invoke({"question": message, "chat_history": chat_history})
            
            # Extract the response
            if isinstance(result, str):
                response = result
            elif hasattr(result, 'content'):
                response = result.content
            else:
                response = str(result) if result is not None else "Could not process the question."
            
            # Update chat history
            self.chat_history.append({"user": message, "assistant": response})
            
            return response
        except Exception as e:
            return f"Error processing message: {str(e)}"