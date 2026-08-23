class VectorStore:
    def __init__(self):
        self.vectors = {}
        self.next_id = 1
    def add(self, vector):
        self.vectors[self.next_id] = vector
        self.next_id += 1
        return self.next_id - 1
    def get(self, vector_id):
        return self.vectors.get(vector_id, None)
    def count(self):
        return len(self.vectors)
