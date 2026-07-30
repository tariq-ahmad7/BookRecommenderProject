from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

Chroma.from_documents(
    documents,
    OpenAIEmbeddings(),
    persist_directory="chroma_db"
)
print("Done — chroma_db folder created.")