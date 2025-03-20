import os
import gradio as gr
from typing import List, Tuple
from dotenv import load_dotenv

from perplexia_ai.week1.factory import Week1Mode, create_chat_implementation1
from perplexia_ai.week2.factory import Week2Mode, create_chat_implementation2

# Load environment variables
load_dotenv()

def create_demo(mode_str: str = "Week1part1"):
    """Create and return a Gradio demo with the specified mode.
    
    Args:
        mode_str: String representation of the mode ('part1', 'part2', or 'part3')
        
    Returns:
        gr.ChatInterface: Configured Gradio chat interface
    """
    # Convert string to enum
    mode_map = {
        "week1_part1": Week1Mode.PART1_QUERY_UNDERSTANDING,
        "week1_part2": Week1Mode.PART2_BASIC_TOOLS,
        "week1_part3": Week1Mode.PART3_MEMORY,
        "week2_part1": Week2Mode.PART1_SearchWeb,
        "week2_part2": Week2Mode.PART2_PolicyRAG,
        "week2_part3": Week2Mode.PART3_CorrectiveRAGlite
    }
    
    if mode_str not in mode_map:
        raise ValueError(f"Unknown mode: {mode_str}. Choose from: {list(mode_map.keys())}")
    
    mode = mode_map[mode_str]
    
    
    # Initialize the chat implementation

    if isinstance(mode, Week1Mode):
        chat_interface = create_chat_implementation1(mode)
    elif isinstance(mode, Week2Mode):
        chat_interface = create_chat_implementation2(mode)
    else:
        raise ValueError("Invalid mode type")



    chat_interface.initialize()
    
    # Create the respond function that uses our chat implementation
    def respond(message: str, history: List[Tuple[str, str]]) -> str:
        """Process the message and return a response.
        
        Args:
            message: The user's input message
            history: List of previous (user, assistant) message tuples
            
        Returns:
            str: The assistant's response
        """
        # Get response from our chat implementation
        return chat_interface.process_message(message, history)
    
    # Create the Gradio interface with appropriate title based on mode
    
    # Create the Gradio interface
    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        theme=gr.themes.Soft()
    )
    
    return demo
