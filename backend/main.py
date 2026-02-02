import os
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Zero-Latency Voice RAG")

@app.on_event("startup")
async def startup_event():
    print("Starting Zero-Latency Voice RAG Engine...")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.websocket("/ws")
async def audio_stream(websocket: WebSocket):
    from backend.stream_manager import StreamManager
    manager = StreamManager(websocket)
    await manager.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
