from typing import Annotated
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import OpenAIEmbeddings

# Load Documents into chromaDB
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from typing import Annotated, Dict, List
from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from perplexia_ai.week2.part1 import UseWebSearch
from perplexia_ai.week2.part2 import PolicyRAG
from typing import List

from typing_extensions import TypedDict

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]  # Changed type to List[Document]

class CorrectiveRAGlite(ChatInterface):
 

    

   
    ### Retrieval Grader

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
        
       
 

    ### Generate

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""),
        ("human", """Question: {question}
Context: {context}
Answer:""")
    ])

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Chain
    rag_chain = prompt | llm | StrOutputParser()


    ### Question Re-writer

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    ### Search
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(k=3)




    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = CorrectiveRAGlite.retriever.invoke(question)
        return {"documents": documents, "question": question}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        
        context = CorrectiveRAGlite.format_docs(documents)
        generation = CorrectiveRAGlite.rag_chain.invoke({"context": context, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = CorrectiveRAGlite.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = CorrectiveRAGlite.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}


    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = CorrectiveRAGlite.web_search_tool.invoke({"query": question})
        web_results = [Document(page_content=d["content"]) for d in docs]  # Convert to Document objects
        documents.extend(web_results)

        return {"documents": documents, "question": question}


    ### Edges


    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"



    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search_node", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)


        
    def initialize(self) -> None:
        PolicyRAG.LoadDocumentsToChromadb(self)
        
        self.graph = CorrectiveRAGlite.workflow.compile()

    def __init__(self):
        self.llm = None
        self.graph = None

    def process_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> str:
        
        events = self.graph.stream(
            {"question": message},
            stream_mode="values",
        )
        response = ""
        for event in events:
            if "generation" in event:  # Changed to look for "generation"
                response = event["generation"]  # Changed to extract from "generation"
                break
        return response