"""Factory for creating Week 1 chat implementations."""

from enum import Enum
from typing import Type

from perplexia_ai.core.chat_interface import ChatInterface

from perplexia_ai.week2.part1 import UseWebSearch
from perplexia_ai.week2.part2 import PolicyRAG
from perplexia_ai.week2.part3 import CorrectiveRAGlite

class Week2Mode(Enum):
    """Modes corresponding to the three parts of Week 1 assignment."""
    PART1_SearchWeb = "part1"  # UseWebSearch
    PART2_PolicyRAG=  "part2"  # PolicyRAG
    PART3_CorrectiveRAGlite = "part3"  # CorrectiveRAGlite


def create_chat_implementation2(mode: Week2Mode) -> ChatInterface:
    """Create and return the appropriate chat implementation.
    
    Args:
        mode: Which part of Week 1 to run
        
    Returns:
        ChatInterface: The appropriate chat implementation
    
    Raises:
        ValueError: If mode is not recognized
    """
    implementations = {
        Week2Mode.PART1_SearchWeb: UseWebSearch,
        Week2Mode.PART2_PolicyRAG: PolicyRAG,
        Week2Mode.PART3_CorrectiveRAGlite: CorrectiveRAGlite

    }
    
    if mode not in implementations:
        raise ValueError(f"Unknown mode: {mode}")
    
    implementation_class = implementations[mode]
    implementation = implementation_class()
    return implementation 