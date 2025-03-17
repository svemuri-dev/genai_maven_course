import os
import sys
import argparse

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Perplexia AI Assistant')
parser.add_argument('--mode', type=str, 
                    default='Week1part1')
args = parser.parse_args()

# Import and run the app
from perplexia_ai.app import create_demo

if __name__ == "__main__":
    demo = create_demo(args.mode)
    demo.launch(show_error=True)
