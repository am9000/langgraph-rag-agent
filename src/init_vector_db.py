from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.notebook import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

# Using local embeddings model downloaded from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

connection = os.getenv("POSTGRES_URL")
collection_name = "csv_docs"

csv_vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)

document_df = pd.read_csv("datasets/documents.csv")

for row in document_df.itertuples(index=False):
    #print(len(row.text))
    text_docs = text_splitter.create_documents([row.text])
    #print(len(text_docs))
    for doc in text_docs:
        print("-" * 20)
        print(doc)
        csv_vector_store.add_documents([doc])
