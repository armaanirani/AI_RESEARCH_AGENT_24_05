from dotenv import load_dotenv
import os

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(
    model='gpt-4.1-nano-2025-04-14',
    api_key=OPENAI_API_KEY
)

# llm = ChatGroq(
#     model='llama-3.3-70b-versatile',
#     api_key=GROQ_API_KEY
# )

# llm = OllamaLLM(
#     model='deepseek-r1:14b'
# )

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

raw_response = agent_executor.invoke({"query": "What is the impact of climate change on global agriculture?"})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output"))
except Exception as e:
    print(f"Error parsing response: {e}")