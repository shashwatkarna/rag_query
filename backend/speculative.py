import asyncio
from rag.retriever import Retriever
from rag.rewriter import QueryRewriter

class SpeculativeEngine:
    def __init__(self):
        self.retriever = Retriever()
        self.rewriter = QueryRewriter()
        self.cache = {} # Map text_hash -> search_results
        self.history = []
        
    async def process_partial(self, partial_text: str):
        """
        Called when ASR gives a confident partial result.
        Trigger a background search.
        """
        # Simple heuristic: only search if > 4 words
        if len(partial_text.split()) < 4:
            return

        print(f"Speculating on: '{partial_text}'")
        
        # 1. Rewrite (Fast)
        rewritten = await self.rewriter.rewrite(partial_text, self.history)
        
        # 2. Search (Parallel)
        results = await self.retriever.search(rewritten, limit=3)
        
        # 3. Cache Result
        self.cache[partial_text] = {
            "rewritten": rewritten,
            "results": results
        }
        print(f"Cached speculative result for: '{partial_text}'")

    async def get_final_result(self, final_text: str):
        """
        Called when ASR gives final result.
        Check cache, or run fresh search.
        """
        print(f"Finalizing: '{final_text}'")
        
        # Check if we have a cached result for a "close enough" partial
        # For now, exact string match or contained substring
        if final_text in self.cache:
            print("CACHE HIT!")
            return self.cache[final_text]
        
        # Fallback: Run fresh
        print("CACHE MISS. Running clean pipeline.")
        rewritten = await self.rewriter.rewrite(final_text, self.history)
        
        # 1. Retrieve (Get more candidates for reranking)
        candidates = await self.retriever.search(rewritten, limit=10)
        
        # 2. Rerank
        from rag.reranker import Reranker
        if not hasattr(self, 'reranker'):
             self.reranker = Reranker()
             
        ranked_results = await self.reranker.rerank(rewritten, candidates, top_k=3)
        
        # Update history
        self.history.append(final_text) 
        
        return {
            "rewritten": rewritten,
            "results": ranked_results
        }
