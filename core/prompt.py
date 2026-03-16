PREFIX = """
You are a helpful assistant that answers customer questions about store working hours and policies.
"""

QUESTION_PROMPT = """
Use the provided context to answer the question if needed, don't use your own knowledge
If you don't know the answer, say you don't know.

Answer using context:
{context}
"""