"""Prompt templates.

Kept in one place because prompts are an experimental variable. If a reviewer asks
whether a finding is prompt-sensitive, the honest answer requires knowing exactly what
was sent, and being able to vary it in one edit.
"""

from __future__ import annotations

SYSTEM_ANSWER = (
    "You are a precise question answering assistant. Answer using only the provided "
    "context. Reply with the answer alone, no explanation. If the context does not "
    "contain the answer, say you cannot determine it."
)

SYSTEM_REWRITE = (
    "You rewrite questions into short search queries. Reply with the query alone."
)

SYSTEM_REFLECT = (
    "You judge whether gathered evidence is sufficient to answer a question. "
    "Reply with exactly one word: SUFFICIENT or INSUFFICIENT."
)

SYSTEM_INSPECT = (
    "You select which document to read in full. Reply with the document number alone."
)


def answer_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def rewrite_prompt(question: str, previous: list[str], evidence_preview: str = "") -> str:
    parts = [f"Original question: {question}"]
    if previous:
        parts.append("Queries already tried:\n" + "\n".join(f"- {q}" for q in previous))
    if evidence_preview:
        parts.append(f"Evidence gathered so far:\n{evidence_preview}")
    parts.append("Write one new search query that would find missing information.")
    return "\n\n".join(parts)


def reflect_prompt(question: str, context: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence gathered:\n{context}\n\n"
        "Is the evidence sufficient to answer the question? "
        "Reply SUFFICIENT or INSUFFICIENT."
    )


def inspect_prompt(question: str, summaries: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Available documents:\n{summaries}\n\n"
        "Which document number should be read in full?"
    )
