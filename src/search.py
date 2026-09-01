from src.similarity import euclidean_dis
class BruteForceSearch:
    def __init__(self, store):
        self.store = store
        # should only set up the BruteForceSearch object.

    def search(self, query):
        dist = float("inf")
        id = None
        for vector_id, vector in self.store.vectors.items():
            d = euclidean_dis(vector, query)
            if d < dist:
                dist = d
                id = vector_id
        return dist, id
