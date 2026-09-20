
class Index:
    def __init__(self, store):
        self.store = store

    def candidates(self, query):
        return list(self.store.vectors.keys())