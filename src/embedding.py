from sentence_transformers import SentenceTransformer
from src.vector import Vector

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        values = self.model.encode(text).tolist()
        return Vector(values)

