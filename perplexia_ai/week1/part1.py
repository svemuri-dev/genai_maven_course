"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.full_chain = None
        self.subchains = {}
    
    def initialize(self) -> None:
        """Initialize components for query understanding."""
        self.llm = ChatOpenAI(model_name="gpt-4", verbose=True)

        template = """Please classify this question: "{question}"
Choose one of these types:
- "Factual" (for "What is...?", "Who invented...?" or any factual statements)
- "Analytical" (for "How does...?", "Why do...?")
- "Calculation" (for mathematical or logical calculations)
- "Comparison" (for "What's the difference between...?")
- "Definition" (for "Define...", "Explain...")

Return only the classification type as a single word."""


        factual_template = """Answers should be concise and direct. Put "Factual" as header 
Question: {question}
Answer:"""
        analytical_template = """Analytical responses should include reasoning steps. Put "Analytical" as header
Question: {question}
Answer:"""
        comparison_template = """Comparisons should use structured formats (tables, bullet points). Put "Comparison" as header
Question: {question}
Answer:"""
        definition_template = """Definitions should include examples and use cases. Put "Definition" as header
Question: {question}
Answer:"""
        calculation_template = """Convert {question} to an expression that a calculator tool can evaluate and give an answer. Return only expression.
Calculator only supports operations +, -, *, /, **

Question: {question}
Answer:"""
        self.full_chain = (
            {"type": classify_chain, "question": lambda x: x["question"]} 
            | RunnableLambda(self.route)
        )
        self.query_classifier_prompt = PromptTemplate.from_template(template)
        classify_chain = (self.query_classifier_prompt | self.llm | StrOutputParser())

        self.subchains = {
            "factual": PromptTemplate.from_template(factual_template) | self.llm,
            "analytical": PromptTemplate.from_template(analytical_template) | self.llm,
            "comparison": PromptTemplate.from_template(comparison_template) | self.llm,
            "definition": PromptTemplate.from_template(definition_template) | self.llm,
            "calculation": PromptTemplate.from_template(calculation_template) | self.llm
        }

    def route(self, info):
        """Route the question to appropriate chain based on classification."""
        chain = self.subchains.get(info["type"].lower(), self.subchains["factual"])
        return chain.invoke({"question": info["question"]})
    
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding."""
        if not message:
            return "Please provide a question to classify."
            
        try:
            result = self.full_chain.invoke({"question": message})
            if isinstance(result, str):
                return result
            if hasattr(result, 'content'):
                return result.content
            return str(result) if result is not None else "Could not process the question."
        except Exception as e:
            return f"Error processing message: {str(e)}"