from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings()


# Step 1: Load and split your documents
def load_doc(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents[0]


mdsplitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
    ],
    strip_headers=False,
)



def mdsplit(file_path, metadata:None):
    doc = load_doc(file_path)
    data = mdsplitter.split_text(doc.page_content)
    if metadata:
        for d in data:
            d.metadata.update(metadata)
    return data

raw_data =[]
raw_data+=mdsplit("./assets/insurance.txt", {"source": "insurance.txt", "domain": "insurance"})
raw_data+=mdsplit("./assets/medical.txt", {"source": "medical.txt", "domain": "medical"})
raw_data+=mdsplit("./assets/legal.txt", {"source": "legal.txt", "domain": "legal"})
raw_data+=mdsplit("./assets/tech.txt", {"source": "tech.txt", "domain": "tech"})


splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    length_function=len
)

chunks = splitter.split_documents(raw_data)



db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="metadata_collection"
)
retreiver = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10,
        "filter": {
            "$or":[
                {"source": "legal.txt"},
                {"domain": "legal"},
                {"h2": "Section"},
                 {"h1": "Section"}
            ]
        }
    })

query = "Tell me about the punishment of murder?"

results = retreiver.invoke(query)



for result in results:
    print(f"Metadata: {result.metadata}")
    print(f"Content: {result.page_content}")
    print("==="*30)
    