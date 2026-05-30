from langchain_community.retrievers import WikipediaRetriever


wiki = WikipediaRetriever(top_k_results=2 , lang="en")

query = "Tell me about the history of artificial intelligence."

docs = wiki.invoke(query)

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Title: {doc.metadata['title']}")
    print(f"Content: {doc.page_content[:200]}...")  # Print the first 200 characters of the content
    print("\n")