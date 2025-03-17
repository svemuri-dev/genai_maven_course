from setuptools import setup, find_packages

setup(
    name="perplexia_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.10",
        "gradio>=4.19.2",
    ],
) 