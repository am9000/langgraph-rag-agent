from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import os
from dotenv import load_dotenv


load_dotenv(override=True)

system_msg = SystemMessage(content="Jesteś asystentem wyszukującym informacje.")
search = DuckDuckGoSearchResults(output_format="list")

# Using local embeddings model downloaded from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

connection = os.getenv("POSTGRES_URL")
collection_name = "csv_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

@tool(response_format="content")
def retrieve(query: str):
    """Wyszukaj informacje o prototypie."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

tools = [search, retrieve]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

llm_with_tools = llm.bind_tools(tools)

def reasoner_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}

tool_node = ToolNode(tools)

builder = StateGraph(MessagesState)

builder.add_node("reasoner", reasoner_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    tools_condition
)
builder.add_edge("tools", "reasoner")

agent_graph = builder.compile()

config = {"configurable": {"thread_id": "abc123"}}

messages = [system_msg]

while True:
    human_message = input("Human Message: ")
    if human_message.lower() == "exit":
        break

    messages.append(HumanMessage(content=human_message))

    output = agent_graph.invoke({"messages": messages}, config)

    messages = output["messages"]
    messages[-1].pretty_print()
