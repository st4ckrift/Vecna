import json
from src.vector import Vector
class VectorStore:
    def __init__(self):
        self.vectors = {}
        self.next_id = 1
        self.documents = {}
    def add(self, vector, text = None):
        self.vectors[self.next_id] = vector
        self.documents[self.next_id] = text
        self.next_id += 1
        return self.next_id - 1
    def get(self, vector_id):
        return self.vectors.get(vector_id, None)
    ## returns vector
    def count(self):
        return len(self.vectors)
    def get_document(self, vector_id):
        return self.documents.get(vector_id, None)
    ## returns original text
    def save(self, filepath):
        vectors = {}
        for vector_id, vector in self.vectors.items():
            vectors[vector_id] = vector.values
        data = {
            "next_id": self.next_id,
            "vectors": vectors[self.next_id],
            "document":self.documents
        }
    def load(self, filepath):
        with open(filepath, "w") as file:
            json.dump(data, file)

