import asyncio
import os
import json
from fastapi import WebSocket
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
from dotenv import load_dotenv

load_dotenv()

class StreamManager:
    def __init__(self, websocket: WebSocket):
        self.fastapi_ws = websocket
        self.dg_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
        self.dg_connection = None

    async def start(self):
        await self.fastapi_ws.accept()
        
        # Initialize Speculative Engine
        from backend.speculative import SpeculativeEngine
        self.engine = SpeculativeEngine()
        
        # Configure Deepgram options
        options = LiveOptions(
            model="nova-2", 
            language="en-US", 
            smart_format=True,
            interim_results=True,
            vad_events=True,
        )
        
        self.dg_connection = self.dg_client.listen.live.v("1")

        # Event Listeners
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)
        
        # Start Deepgram connection
        print("Starting Deepgram connection...")
        if self.dg_connection.start(options) is False:
             print("Deepgram failed to start")
             return
        print("Deepgram connection started successfully")

        # Start receiving audio from client
        try:
            print("Listening for audio bytes from client...")
            chunk_count = 0
            while True:
                data = await self.fastapi_ws.receive_bytes()
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"Received audio chunk #{chunk_count}, size: {len(data)}")
                self.dg_connection.send(data)
        except Exception as e:
            print(f"WebSocket closed: {e}")
            await self.stop()

    def on_transcript(self, result, **kwargs):
        asyncio.create_task(self._process_transcript(result))

    async def _process_transcript(self, result):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return
            
        is_final = result.is_final
        speech_final = result.speech_final
        
        # Speculative Logic
        if is_final:
            print(f"FINAL: {sentence}")
            
            # Send Filler immediately
            from backend.filler import FillerGenerator
            filler = FillerGenerator().get_filler()
            await self.fastapi_ws.send_text(json.dumps({
                "type": "filler",
                "text": filler
            }))
            
            rag_result = await self.engine.get_final_result(sentence)
            print(f"RAG RESULT: {rag_result}")
            
            # 1. Voice Optimization
            from voice.processor import VoiceProcessor
            if not hasattr(self, 'processor'):
                self.processor = VoiceProcessor()
                
            top_answer = rag_result["results"][0] if rag_result["results"] else "I couldn't find that information in the manual."
            spoken_text = await self.processor.to_spoken_english(top_answer)
            print(f"SPOKEN: {spoken_text}")
            
            # 2. TTS Generation
            from voice.tts import TTSClient
            if not hasattr(self, 'tts'):
                self.tts = TTSClient()
                
            audio_bytes = await self.tts.generate_audio(spoken_text)
            
            # Send Metadata and Audio
            await self.fastapi_ws.send_text(json.dumps({
                "type": "final_result",
                "text": sentence,
                "rag": rag_result,
                "spoken_text": spoken_text
            }))
            await self.fastapi_ws.send_bytes(audio_bytes)
        else:
            print(f"PARTIAL: {sentence}")
            await self.engine.process_partial(sentence)

    def on_error(self, error, **kwargs):
        print(f"Deepgram Error: {error}")

    async def stop(self):
        if self.dg_connection:
            self.dg_connection.finish()
            self.dg_connection = None
