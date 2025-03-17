from typing import Annotated, Dict, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langgraph.graph import StateGraph, START, END
from perplexia_ai.core.chat_interface import ChatInterface

# Define state for application
class State(TypedDict):
    question: str
    documents: List[Document]
    answer: str


class PolicyRAG(ChatInterface):
    # Define application steps
    def retrieve(self, state: State):
        retrieved_docs = self.semantic_chunk_vectorstore.similarity_search(state["question"], k=2)
        return {"documents": retrieved_docs}
    
    def extract_sources(self, documents):
        """Extract unique sources from documents and format them for display"""
        sources = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"Document {i}")
            if source not in sources:
                sources.append(source)
        
        formatted_sources = "\n\nSources:\n" + "\n".join([f"[{i}] {source}" for i, source in enumerate(sources)])
        return formatted_sources

    def generate(self, state: State):
        # Create a list of formatted documents with citations
        formatted_docs = []
        for i, doc in enumerate(state["documents"]):
            source = doc.metadata.get("source", f"Document {i}")
            # Format each document with a citation marker
            formatted_docs.append(f"Document [{i}]: {doc.page_content}\nSource: {source}")
        
        docs_content = "\n\n".join(formatted_docs)
        
        system_prompt = """Answer the question based on the provided documents.
        If the answer is not found in the documents, say 'I can't answer the question'.
        Please cite your sources using the document numbers provided in square brackets, 
        for example [0], [1], etc.
        
        Documents:
        {documents}
        
        Question: {question}
        """

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        # Invoke the model with formatted prompt
        response = self.llm.invoke(
            prompt_template.format_messages(
                question=state["question"], 
                documents=docs_content
            )
        )
        
        sources_text = self.extract_sources(state["documents"])
        answer = response.content + sources_text
        return {"answer": answer}
    
    def initialize(self) -> None:
        loader = PyPDFDirectoryLoader("D:\Learn\genai_maven_course\perplexia_ai\RAGDataset")
        documents = loader.load()

        if not documents:
            raise ValueError("No documents were loaded from the directory")
        
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="gradient")
        documents = text_splitter.split_documents(documents)
        self.semantic_chunk_vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings()) 

        # Compile application and test
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
    
        self.graph = graph_builder.compile()

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
            if "answer" in event:
                response = event["answer"]
                break
        return response