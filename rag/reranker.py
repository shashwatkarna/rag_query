from sentence_transformers import CrossEncoder
import asyncio
import functools

class Reranker:
    def __init__(self):
        print("Initializing Reranker (ms-marco-MiniLM-L-6-v2)...")
        # fast and decent accuracy
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    async def rerank(self, query: str, docs: list[str], top_k: int = 3):
        if not docs:
            return []
            
        # CrossEncoder expects pairs of (query, doc)
        pairs = [[query, doc] for doc in docs]
        
        # Run in thread pool to avoid blocking async event loop
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, 
            functools.partial(self.model.predict, pairs)
        )
        
        # Sort by score desc
        results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top_k docs
        return [doc for doc, score in results[:top_k]]
