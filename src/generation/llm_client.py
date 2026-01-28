import os
import logging
from typing import List
from groq import Groq

logger = logging.getLogger("llm-client")


SYSTEM_PROMPT = """
You are a university teaching assistant.

Rules:
- Answer the question using ONLY the provided context.
- You may rephrase or translate the information to match the language of the user's question.
- Do NOT add new information.
- Do NOT use external knowledge.
- When you state a fact, add the reference number in square brackets, e.g. [1], [2].
- If the answer is not explicitly present, say:
  "Not found in the provided lecture material."
"""


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        self.model = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

        self.client = Groq(api_key=api_key)

    async def generate(
        self,
        question: str,
        context_chunks: List[str],
        chat_history: List[str] | None = None
    ) -> str:
        context = "\n\n".join(context_chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.exception("Groq LLM call failed")
            raise RuntimeError("LLMGenerationError") from e
