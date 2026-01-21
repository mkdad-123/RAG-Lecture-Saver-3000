from generation.llm_client import LLMClient

llm = LLMClient()

FAITHFULNESS_PROMPT = """
Given the following CONTEXT and ANSWER:

Determine whether the ANSWER is fully supported by the CONTEXT.
Reply ONLY with one word:
- YES
- NO
"""

def faithfulness_score(context: str, answer: str)-> int:

    response = llm.generate(
        question=FAITHFULNESS_PROMPT,
        context_chunks=[f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"]
    )
    
    return 1 if response.strip().upper().startswith("YES") else 0 