import os 
from groq import Groq
from typing import List
from dotenv import load_dotenv

class LLMClient:
    def __init__(self):
        load_dotenv() 
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL")


    def generate(
            self,
            question:str, 
            context_chunks: List[str] , 
            chat_history: List[str]| None = None
            )-> str:
        context = "\n\n".join(context_chunks)    

        history = ""
        if chat_history:
            history = "\n".join(chat_history)

        SYSTEM_PROMPT = """
You are a university teaching assistant.

Rules:
- Answer the question using ONLY the provided context.
- You may rephrase or translate the information to match the language of the user's question.
- Do NOT add new information.
- Do NOT use external knowledge.
-When you state a fact, add the reference number in square brackets, e.g. [1], [2].
- If the answer is not explicitly present, say:
  "Not found in the provided lecture material."
"""



        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
            Context:
            {context}

            Question:
            {question}
            """}
        ]


        
        response = self.client.chat.completions.create(
            model = self.model,
            messages= messages,
            temperature=0.5,
            max_tokens=10000
        )

        return response.choices[0].message.content.strip()
