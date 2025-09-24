from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(
    api_key="ollama",
    model="qwen3:0.6b",
    base_url="http://localhost:11434/v1",
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            First, use the necessary tools to gather information for the user query.
            Then, summarize your findings in a coherent and extensive manner, with a minimum of 2000 words for your summary.
            Finally, provide the output in the following JSON format and do not include any other text:

            {format_instructions}

            For the "tools_used" field, list only the names of the tools you used, such as "search", "wiki", or "save_text_to_file".
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
llm = llm.bind_tools(tools)
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    text = raw_response["output"]
    structured_response = parser.parse(text)
    print(structured_response)
except Exception as e:
    print("end of response")