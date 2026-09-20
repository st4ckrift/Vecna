import json
from src.vector import Vector

class VectorStore:
    def __init__(self):
        self.vectors = {}
        self.next_id = 1
        self.documents = {}

    def add(self, vector, text=None):
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
            "vectors": vectors,
            "documents": self.documents
        }

        with open(filepath, "w") as file:
            json.dump(data, file)

    def load(self, filepath):
        with open(filepath, "r") as file:
            data = json.load(file)

        self.next_id = data["next_id"]

        self.vectors = {}
        self.documents = {}

        for vector_id, values in data["vectors"].items():
            vector_id = int(vector_id)
            self.vectors[vector_id] = Vector(values)

        for vector_id, document in data["documents"].items():
            vector_id = int(vector_id)
            self.documents[vector_id] = document