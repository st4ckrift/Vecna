from src.similarity import euclidean_dis
class BruteForceSearch:
    def __init__(self, store, index = None):
        self.store = store
        self.index = index
        # should only set up the BruteForceSearch object.

    def search(self, query, k=1, filters=None):
        if self.index is not None:
            candidates = self.index.candidates(query)
        else:
            candidates = list(self.store.vectors.keys())
    
        dist_list = []
        for vector_id in candidates:
            vector = self.store.get(vector_id)
            if vector is None:
                continue
            metadata = self.store.get_metadata(vector_id)
            if filters and any(metadata.get(key) != value for key, value in filters.items()):
                continue
            d = euclidean_dis(vector, query)
            dist_list.append((vector_id, d))
        sorted_list = sorted(dist_list, key = lambda x: x[1])
        return sorted_list[:k]

    def search_text(self, text, embedder, k=1, filters=None):
        query_vector = embedder.embed(text)
        results = self.search(query_vector, k, filters=filters)
        text_results = []
        for vector_id, distance in results:
            document = self.store.get_document(vector_id)
            text_results.append((document, distance))
        return text_results


