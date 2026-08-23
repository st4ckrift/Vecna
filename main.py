from src.vector import Vector
from src.similarity import euclidean_dis
from src.storage import VectorStore
store = VectorStore()

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

id1 = store.add(v1)
id2 = store.add(v2)

print(id1)
print(id2)

print(store.get(id1).values)
print(store.get(id2).values)

print(store.count())
print(store.get(999))

print(euclidean_dis(v1,v2))

