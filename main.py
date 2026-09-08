from src.embedding import Embedder
from src.storage import VectorStore
from src.search import BruteForceSearch

embedder = Embedder()
store = VectorStore()

documents = [
    "Python is a programming language",
    "Python is commonly used for software development",
    "Bananas are a type of fruit"
]

for text in documents:
    vector = embedder.embed(text)
    store.add(vector, text)

searcher = BruteForceSearch(store)

results = searcher.search_text(
    "programming with Python",
    embedder,
    k=2
)

print(results)
