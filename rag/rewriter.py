import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class QueryRewriter:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant" # Updated to supported model

    async def rewrite(self, query: str, history: list[str]) -> str:
        """
        Rewrites the latest query based on conversation history to resolve coreferences.
        """
        if not history:
            return query
            
        system_prompt = """You are a query rewriting engine. Your job is to rewrite the LAST user query to be standalone, resolving any pronouns (it, they, the first one) using the conversation history.
        Output ONLY the rewritten query. Do not explain.
        
        Example:
        History: ["How much is the X100?", "The X100 costs $500."]
        User: "What is its battery life?"
        Rewritten: "What is the battery life of the X100?"
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"History: {history}\nUser: {query}"}
        ]
        
        try:
            # We use synchronous call here as Groq is extremely fast, 
            # but in production you might want to wrap this in run_in_executor
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0,
                max_tokens=50,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Rewriter Error: {e}")
            return query
