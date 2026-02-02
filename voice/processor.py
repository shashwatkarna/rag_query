import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class VoiceProcessor:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    async def to_spoken_english(self, text: str) -> str:
        """
        Rewrites complex technical text into simple, conversational spoken English.
        """
        system_prompt = """You are a Voice AI formatter. Your goal is to rewrite the input text for Text-to-Speech synthesis.
        Rules:
        1. Remove all Markdown (*, #, [], links).
        2. Keep sentences short and punchy.
        3. Expand abbreviations (e.g., "AI" -> "A-I", "200MB" -> "200 megabytes").
        4. Use phonetic spelling for difficult technical terms if needed.
        5. Maintain the core technical accuracy but explain it simply.
        6. Start directly with the answer.
        
        Input: "The **X-200** requires 5V/2A input."
        Output: "The X-200 requires five volts and two amps of input."
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=150,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Voice Processor Error: {e}")
            return text
