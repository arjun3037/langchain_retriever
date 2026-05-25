from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vectorstore = Chroma.from_documents(
    documents, 
    embeddings, 
    collection_name="my_collection"
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is LangChain?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print("\n")
