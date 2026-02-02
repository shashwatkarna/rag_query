import asyncio
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Mock WebSocket
class MockWebSocket:
    async def send_text(self, text):
        print(f"[WS] Sent Text: {text[:100]}...")
    async def send_bytes(self, b):
        print(f"[WS] Sent Audio Bytes: {len(b)} bytes")

async def test_pipeline():
    print("--- Starting Zero-Latency Pipeline Verification ---")
    
    # 1. Initialize Components
    print("Initializing Engine...")
    t0 = time.time()
    try:
        from backend.speculative import SpeculativeEngine
        engine = SpeculativeEngine()
        
        from voice.processor import VoiceProcessor
        processor = VoiceProcessor()
        
        from voice.tts import TTSClient
        tts = TTSClient()
        print(f"Initialization took {time.time() - t0:.2f}s")

        # --- WARMUP START ---
        print("\nWarming up models (loading Reranker into memory)...")
        # Run a dummy query to force model loading
        await engine.get_final_result("warmup")
        print("Warmup complete.")
        # --- WARMUP END ---

        # 2. Simulate User Query
        query = "What is the password policy?"
        print(f"\nQuery: '{query}'")

        # 3. Measure RAG Latency
        t1 = time.time()
        print("Executing RAG...")
        # Simulate final result path in stream_manager
        rag_result = await engine.get_final_result(query)
        rag_time = time.time() - t1
        print(f"RAG (Vector + Rerank) Time: {rag_time:.4f}s")
        
        if rag_result["results"]:
            print(f"Top Result: {rag_result['results'][0][:100]}...")
        else:
            print("No results found. (Ingestion might be incomplete)")

        # 4. Measure Voice Processing Latency
        t2 = time.time()
        top_answer = rag_result["results"][0] if rag_result["results"] else "No info found."
        spoken = await processor.to_spoken_english(top_answer)
        proc_time = time.time() - t2
        print(f"Voice Processing Time: {proc_time:.4f}s")
        print(f"Spoken Text: {spoken}")

        # 5. Measure TTS Latency (TTFB - Time to First Byte)
        t3 = time.time()
        print("Streaming TTS (HTTP) to measure TTFB...")
        
        first_chunk_received = False
        ttfb = 0
        total_audio_size = 0
        
        # Use HTTP Streaming for reliable verification
        async for chunk in tts.generate_audio_stream(spoken):
            if not first_chunk_received:
                ttfb = time.time() - t3
                first_chunk_received = True
                print(f"TTS TTFB: {ttfb:.4f}s")
            total_audio_size += len(chunk)

        print(f"Total TTS Stream Time: {time.time() - t3:.4f}s")
        print(f"Total Audio Size: {total_audio_size} bytes")

        # TTFB is the critical metric for "Zero Latency" perception
        # We assume RAG latency is hidden by speculative execution in the real app, 
        # but here we sum them up. However, Voice+TTS TTFB is what matters for "Response Start".
        voice_plus_tts_ttfb = proc_time + ttfb
        print(f"\nVoice+TTS Latency (Reaction Time): {voice_plus_tts_ttfb:.4f}s")
        
        total_ttfb = rag_time + proc_time + ttfb
        print(f"Total Pipeline Latency (TTFB): {total_ttfb:.4f}s")
        
        if total_ttfb < 1.0:
            print("✅ SUCCESS: Low Latency (< 1.0s)")
        else:
            print(f"⚠️ Latency: {total_ttfb:.2f}s")
            
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
