from src.similarity import euclidean_dis
class BruteForceSearch:
    def __init__(self, store):
        self.store = store
        # should only set up the BruteForceSearch object.

    def search(self, query, k = 1):
        dist_list = []
        for vector_id, vector in self.store.vectors.items():
            d = euclidean_dis(vector, query)
            dist_list.append((vector_id, d))
        sorted_list = sorted(dist_list, key = lambda x: x[1])
        return sorted_list[:k]
