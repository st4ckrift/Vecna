from src.vector import Vector
from src.storage import VectorStore
from src.embedding import Embedder
from src.search import BruteForceSearch


# Create embedder and store
embedder = Embedder()
store = VectorStore()


# Add documents
documents = [
    "Python is a programming language",
    "Python is commonly used for software development",
    "Bananas are a type of fruit"
]

for text in documents:
    vector = embedder.embed(text)
    store.add(vector, text)


# Save the store
store.save("vectors.json")

print("Store saved.")
print("Number of vectors:", store.count())


# Simulate a new program/run
new_store = VectorStore()

# Load previously saved data
new_store.load("vectors.json")

print("\nStore loaded.")
print("Number of vectors:", new_store.count())


# Check loaded document
print("Document 1:", new_store.get_document(1))

# Check loaded vector
print("Vector 1:", new_store.get(1).values[:5])


# Search using the loaded store
searcher = BruteForceSearch(new_store)

results = searcher.search_text(
    "programming with Python",
    embedder,
    k=2
)

print("\nSearch results:")
for document, distance in results:
    print(document)
    print("Distance:", distance)


# Check that IDs continue correctly
new_id = new_store.add(
    embedder.embed("Python has many libraries"),
    "Python has many libraries"
)

print("\nNew vector ID:", new_id)