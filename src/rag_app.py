from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

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

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

summarize_system_prompt_str = (
    "Przeanalizuj historię konwersacji oraz ostatnie pytanie użytkownika."
    "Na tej podstawie sformułuj zwięzłe pytanie, które będzie równoważne "
    "pytaniu użytkownika, ale nie będzie wymagało całej długiej historii "
    "konwersacji. Nie odpowiadaj na pytanie, tylko wygeneruj je w zwięzłej formie."
)

summarize_system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summarize_system_prompt_str),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    summarize_system_prompt
)

chat_system_prompt_str = (
    "Jesteś asystentem odpowiadającym na pytania. Przeanalizuj kontekst "
    "i odpowiedz zwiężle na pytanie użytkownika. Jeśli nie możesz znaleźć odpowiedzi "
    "to odpowiedz 'Nie mogę znaleźć odpowiedzi.'"
    "\n\n"
    "{context}"
)

chat_system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chat_system_prompt_str),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chat_chain = create_stuff_documents_chain(llm, chat_system_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, chat_chain)

chat_history = []

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    print(f"AI: {result['answer']}")
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))
