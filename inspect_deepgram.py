
import deepgram
import inspect
import pkgutil

print(f"Deepgram file: {deepgram.__file__}")
print(f"Dir(deepgram): {dir(deepgram)}")

try:
    from deepgram import LiveTranscriptionEvents
    print("Succcess: from deepgram import LiveTranscriptionEvents")
except ImportError as e:
    print(f"Fail: from deepgram import LiveTranscriptionEvents -> {e}")

try:
    import deepgram.clients
    print("Success: import deepgram.clients")
except ImportError as e:
    print(f"Fail: import deepgram.clients -> {e}")

try:
    from deepgram.clients.live.v1 import LiveTranscriptionEvents
    print("Success: from deepgram.clients.live.v1 import LiveTranscriptionEvents")
except ImportError as e:
    print(f"Fail: from deepgram.clients.live.v1 import LiveTranscriptionEvents -> {e}")

# Try to find where it is
def find_live_options(module, path="deepgram"):
    if hasattr(module, "LiveTranscriptionEvents"):
        print(f"FOUND LiveTranscriptionEvents in {path}")
        return
    
    if hasattr(module, "__path__"):
        for _, name, ispkg in pkgutil.iter_modules(module.__path__):
            if name.startswith("_"): continue
            try:
                submod = __import__(f"{module.__name__}.{name}", fromlist=[""])
                # Recursion might be too much, just check one level
                if hasattr(submod, "LiveTranscriptionEvents"):
                     print(f"FOUND LiveTranscriptionEvents in {path}.{name}")
            except Exception:
                pass

find_live_options(deepgram)
