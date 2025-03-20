from typing import Annotated, Dict, List
from typing_extensions import TypedDict
import os

from pydantic import BaseModel, Field
from perplexia_ai.core.chat_interface import ChatInterface

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


class CorrectiveRAGlite(ChatInterface):
    """RAG implementation with document relevance grading and web search fallback."""
    
    def __init__(self, documents_path="D:\\Learn\\Repos\\genai_maven_course\\perplexia_ai\\RAGDataset"):
        self.documents_path = documents_path
        
    def initialize(self) -> None:
        """Initialize all components and build the workflow."""
        # Setup document store
        self.setup_document_store()
        
        # Setup LLMs and tools
        self.setup_llms()
        self.setup_tools()
        
        # Setup chains
        self.setup_grading_chain()
        self.setup_generation_chain()
        self.setup_rewriting_chain()
        
        # Build and compile workflow
        self.build_workflow()
        
    def setup_document_store(self):
        """Load documents into ChromaDB."""
        try:
            loader = PyPDFDirectoryLoader(self.documents_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No documents were loaded from the directory")
            
            embeddings = OpenAIEmbeddings()
            text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="gradient")
            documents = text_splitter.split_documents(documents)
            
            self.document_store = Chroma.from_documents(documents=documents, embedding=embeddings)
            self.retriever = self.document_store.as_retriever()
        except Exception as e:
            raise ValueError(f"Failed to load documents: {str(e)}")
    
    def setup_llms(self):
        """Setup language models."""
        self.llm_grader = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)  # Updated model name
        self.llm_generator = ChatOpenAI(model="gpt-4", temperature=0)
        self.llm_rewriter = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Setup structured output with function calling method
        self.structured_llm_grader = self.llm_grader.with_structured_output(
            GradeDocuments,
            method="function_calling"  # Explicitly set the method
        )
    
    def setup_tools(self):
        """Setup search tools."""
        self.web_search_tool = TavilySearchResults(k=3)
    
    def setup_grading_chain(self):
        """Setup document grading chain."""
        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader
    
    def setup_generation_chain(self):
        """Setup answer generation chain."""
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""),
            ("human", """Question: {question}
            Context: {context}
            Answer:""")
        ])
        
        self.rag_chain = generation_prompt | self.llm_generator | StrOutputParser()
    
    def setup_rewriting_chain(self):
        """Setup question rewriting chain."""
        system_prompt = """You a question re-writer that converts an input question to a better version that is optimized
            for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        
        self.question_rewriter = rewrite_prompt | self.llm_rewriter | StrOutputParser()
    
    def build_workflow(self):
        """Build the workflow graph."""
        self.workflow = StateGraph(GraphState)
        
        # Add nodes - change the web_search node name to avoid conflict
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("transform_query", self.transform_query)
        self.workflow.add_node("web_search_node", self.web_search)  # Changed node name
        
        # Connect nodes
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        # Update the edge connections to use the new node name
        self.workflow.add_edge("transform_query", "web_search_node")
        self.workflow.add_edge("web_search_node", "generate")
        self.workflow.add_edge("generate", END)
        
        # Compile the graph
        self.graph = self.workflow.compile()
    
    def process_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Process incoming messages and return responses."""
        events = self.graph.stream(
            {"question": message},
            stream_mode="values",
        )
        for event in events:
            if "generation" in event:
                return event["generation"]
        return ""
    
    def retrieve(self, state: Dict) -> Dict:
        """Retrieve relevant documents."""
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def grade_documents(self, state: Dict) -> Dict:
        """Grade document relevance."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        filtered_docs = []
        web_search = "No"
        
        for doc in state["documents"]:
            score = self.retrieval_grader.invoke({
                "question": question, 
                "document": doc.page_content
            })
            
            if score.binary_score == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                
        return {
            "documents": filtered_docs, 
            "question": question, 
            "web_search": web_search
        }
    
    @staticmethod
    def format_docs(docs):
        """Format documents for processing."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate(self, state: Dict) -> Dict:
        """Generate answer based on documents."""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        context = self.format_docs(documents)
        generation = self.rag_chain.invoke({"context": context, "question": question})
        
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation
        }
    
    def transform_query(self, state: Dict) -> Dict:
        """Transform the query to produce a better question."""
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        
        better_question = self.question_rewriter.invoke({"question": question})
        
        return {
            "documents": documents, 
            "question": better_question
        }
    
    def web_search(self, state: Dict) -> Dict:
        """Perform web search based on the re-phrased question."""
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        
        docs = self.web_search_tool.invoke({"query": question})
        web_results = [Document(page_content=d["content"]) for d in docs]
        documents.extend(web_results)
        
        return {
            "documents": documents, 
            "question": question
        }
    
    def decide_to_generate(self, state: Dict) -> str:
        """Determines whether to generate an answer or re-generate a question."""
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        
        if web_search == "Yes":
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"