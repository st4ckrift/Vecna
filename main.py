from src.vector import Vector
from src.similarity import euclidean_dis

v1 = Vector([1,2])
print(v1.values)

v2 = Vector([1,2,3])
print(v2.values)

print(euclidean_dis(v1,v2))

