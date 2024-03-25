import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=["My name is chayan", "My name is not chayan"],
    metadatas=[{"source": "My name is true"}, {"source": "My name is not true"}],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts= ['what is my name?'],
    n_results= 2
)
print(results)