"""
prompts.py
----------
Prompt template used to ground the LLM's answers strictly in the
retrieved documentation chunks (Retrieval-Augmented Generation).
"""

from typing import List

FALLBACK_ANSWER = "I couldn't find this information in the uploaded documentation."

SYSTEM_INSTRUCTIONS = (
    "You are an Enterprise AI Documentation Assistant.\n"
    "Answer only using the provided documentation.\n"
    f'If information is unavailable, respond: "{FALLBACK_ANSWER}"'
)


def build_prompt(question: str, chunks: List[str]) -> str:
    """
    Assemble the final prompt sent to the LLM, combining the system
    instructions, retrieved context chunks, and the user's question.
    """
    context = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "Context\n"
        "----------------\n"
        f"{context if context else '(no context retrieved)'}\n"
        "----------------\n\n"
        "Question\n"
        f"{question}\n\n"
        "Answer"
    )
    return prompt


def build_sample_questions_prompt(chunks: List[str], num_questions: int) -> str:
    """
    Assemble a prompt asking the LLM to write example questions that are
    genuinely answerable from the given documentation excerpts.

    Used to populate the UI's "Sample questions" section with suggestions
    grounded in whatever the user actually uploaded, instead of a static,
    unrelated list.
    """
    context = "\n\n".join(
        f"[Excerpt {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    prompt = (
        "You are helping developers explore a set of technical documentation.\n"
        f"Based only on the excerpts below, write exactly {num_questions} short, "
        "natural language questions that a developer could ask and have answered "
        "using this documentation.\n"
        "Rules:\n"
        "- One question per line.\n"
        "- No numbering, bullets, or extra commentary.\n"
        "- Do not invent details that aren't supported by the excerpts.\n\n"
        "Documentation Excerpts\n"
        "----------------\n"
        f"{context}\n"
        "----------------\n\n"
        "Questions:"
    )
    return prompt
