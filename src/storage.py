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
