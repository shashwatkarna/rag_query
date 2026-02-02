import os
import json
import base64
import websockets
from dotenv import load_dotenv

load_dotenv()

class TTSClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.url = "wss://api.deepgram.com/v1/peak?model=aura-asteria-en" # Aura model

    async def stream_audio(self, text_stream):
        """
        Connects to Deepgram TTS and yields audio bytes.
        text_stream: async generator yielding text chunks.
        """
        headers = {"Authorization": f"Token {self.api_key}"}
        
        async with websockets.connect(self.url, extra_headers=headers) as ws:
            # Send text
            await ws.send(json.dumps({"text": text_stream})) 
            # Note: Deepgram TTS WS might require specific flush logic or just raw string sending.
            # For simplicity, we'll assume we send the whole text for this MVP integration
            # or we iterate if text_stream is an iterator.
            
            await ws.send(json.dumps({"type": "Close"}))
            
            while True:
                try:
                    msg = await ws.recv()
                    # Deepgram sends audio in binary messages?
                    # Needs verification of specific API protocol. 
                    # Assuming standard binary response for now.
                    yield msg 
                except websockets.exceptions.ConnectionClosed:
                    break
    
    # Simpler HTTP version for MVP (lower complexity than WS for TTS if not strict about <200ms)
    # But user wants <800ms TTFB.
    # Deepgram Aura via REST is also very fast. 
    # Let's use REST for simplicity to avoid WS-in-WS complexity for now, unless latency is bad.
    
    async def generate_audio(self, text: str):
        import httpx
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json={"text": text})
            return response.content

    async def generate_audio_stream(self, text: str):
        """
        Streams audio via HTTP (Chunked) for TTFB measurement.
        """
        import httpx
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json={"text": text}) as response:
                async for chunk in response.aiter_bytes():
                     yield chunk
