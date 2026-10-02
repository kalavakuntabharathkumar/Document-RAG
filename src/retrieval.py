from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self, records: list[dict]):
        if not records:
            raise ValueError("No text chunks found. Add PDF files to data/pdfs.")
        self.records = records
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in records])

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {**self.records[i], "score": float(scores[i])}
            for i in ranked if scores[i] > 0
        ]
