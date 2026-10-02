from pathlib import Path
from .ingest import load_corpus
from .retrieval import TfidfRetriever

PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"

# Expected source file for each question. The evaluation is intentionally
# transparent and checks source-document recall@3, not medical answer quality.
QUESTIONS = [
    ("What is hypertension?", "01_blood_pressure.pdf"),
    ("Which lifestyle habits support healthy blood pressure?", "01_blood_pressure.pdf"),
    ("What does salt have to do with blood pressure?", "01_blood_pressure.pdf"),
    ("How can people monitor blood pressure?", "01_blood_pressure.pdf"),
    ("What is physical activity?", "02_physical_activity.pdf"),
    ("How much activity should adults aim for?", "02_physical_activity.pdf"),
    ("What are examples of moderate activity?", "02_physical_activity.pdf"),
    ("Why is sitting for long periods discouraged?", "02_physical_activity.pdf"),
    ("What is a balanced diet?", "03_healthy_eating.pdf"),
    ("Why are fruits and vegetables important?", "03_healthy_eating.pdf"),
    ("What are whole grains?", "03_healthy_eating.pdf"),
    ("How can people reduce excess sugar?", "03_healthy_eating.pdf"),
    ("What is sleep hygiene?", "04_sleep_health.pdf"),
    ("Why is a regular sleep schedule useful?", "04_sleep_health.pdf"),
    ("What habits can support better sleep?", "04_sleep_health.pdf"),
    ("What is hand hygiene?", "05_hand_hygiene.pdf"),
    ("When should hands be cleaned?", "05_hand_hygiene.pdf"),
    ("Why is handwashing important?", "05_hand_hygiene.pdf"),
    ("What is stress management?", "06_stress_wellbeing.pdf"),
    ("Name some healthy ways to manage stress.", "06_stress_wellbeing.pdf"),
]

def evaluate(chunk_words: int) -> tuple[int, int]:
    records = load_corpus(PDF_DIR, chunk_words=chunk_words)
    retriever = TfidfRetriever(records)
    hits = sum(
        any(p["source"] == expected for p in retriever.search(question, top_k=3))
        for question, expected in QUESTIONS
    )
    return hits, len(QUESTIONS)

if __name__ == "__main__":
    for size in (200, 500):
        hits, total = evaluate(size)
        print(f"{size}-word chunks: {hits}/{total} questions retrieved expected source in top 3")
