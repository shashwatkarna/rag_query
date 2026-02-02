import asyncio
import os
import json
from deepgram import DeepgramClient
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
        
        # Start Deepgram connection
        print("Starting Deepgram connection...")
        try:
            # New SDK Pattern: Usage connection context manager
            # Manually enter context to keep connection alive
            connection_ctx = self.dg_client.listen.live.v("1").connect(
                model="nova-2",
                language="en-US",
                smart_format="true",
                interim_results="true",
                vad_events="true",
            )
            # Enter async context
            self.dg_connection = await connection_ctx.__aenter__()
            
            # Register Listeners
            from deepgram.core.events import EventType
            self.dg_connection.on(EventType.MESSAGE, self.on_message)
            self.dg_connection.on(EventType.ERROR, self.on_error)
            
            # Start background listener task
            # The SDK's start_listening() loops forever, so we need to run it in background
            self.listener_task = asyncio.create_task(self.dg_connection.start_listening())
            
            print("Deepgram connection started successfully")

            # Start receiving audio from client
            print("Listening for audio bytes from client...")
            chunk_count = 0
            while True:
                data = await self.fastapi_ws.receive_bytes()
                chunk_count += 1
                if chunk_count % 50 == 0:
                     print(f"Received audio chunk #{chunk_count}, size: {len(data)}")
                await self.dg_connection.send(data)

        except Exception as e:
            print(f"WebSocket closed: {e}")
            await self.stop()

    def on_message(self, result, **kwargs):
        # Result is likely a ListenV1ResultsEvent or similar Pydantic model
        # We need to check if it has a transcript
        # The structure is result.channel.alternatives[0].transcript
        
        # Note: 'result' might be MetadataEvent or UtteranceEndEvent too.
        # We need to check carefully.
        try:
             # Check if it's a result with a channel
            if hasattr(result, 'channel'):
                 asyncio.create_task(self._process_transcript(result))
        except Exception:
            pass

    async def _process_transcript(self, result):
        # Access pydantic model fields
        if not result.channel.alternatives:
             return
             
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return
            
        is_final = result.is_final
        # speech_final might not be directly on result in V1? 
        # Checking schema... result.speech_final maps to 'speech_final' prop
        speech_final = getattr(result, 'speech_final', False)
        
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
        # If we manually entered context, we must exit it?
        # Or just close connection?
        pass # Context manager nuances... let's just let it die for now or implement strict cleanup later

