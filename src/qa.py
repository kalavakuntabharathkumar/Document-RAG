from openai import OpenAI

def answer_question(question: str, passages: list[dict], model: str = "gpt-4o-mini") -> str:
    if not passages:
        return "I couldn't find relevant information in the indexed documents."
    context = "\n\n".join(
        f"[{p['source']} | chunk {p['chunk_id']}] {p['text']}"
        for p in passages
    )
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Answer using only the supplied context. If the context does not "
                "contain enough evidence, say you cannot determine the answer "
                "from these documents. Do not add outside facts."
            )},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
        ],
    )
    return response.choices[0].message.content or "No answer was returned."
