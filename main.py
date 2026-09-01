from src.vector import Vector
from src.similarity import euclidean_dis
from src.storage import VectorStore
from src.search import BruteForceSearch

store = VectorStore()

store.add(Vector([1, 2]))
store.add(Vector([10, 10]))
store.add(Vector([2, 3]))

searcher = BruteForceSearch(store)
print(searcher.search(Vector([1, 1]), 10))


