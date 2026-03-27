
import wikipediaapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

class WikiRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.wiki = wikipediaapi.Wikipedia('en')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query: str) -> List[str]:
        try:
            page = self.wiki.page(query)
            if not page.exists():
                # Fallback: search
                search_results = self.wiki.search(query, results=1)
                if not search_results:
                    return ["No relevant information found."]
                page = self.wiki.page(search_results[0])
            
            content = page.text[:2000]  # limit length
            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 50]
            
            if not paragraphs:
                return ["No substantial content found."]
            
            # Re-rank paragraphs by similarity to query
            query_emb = self.encoder.encode(query, convert_to_tensor=True)
            para_embs = self.encoder.encode(paragraphs, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, para_embs)[0]
            top_idx = np.argsort(scores.cpu().numpy())[::-1][:self.top_k]
            return [paragraphs[i] for i in top_idx]
        
        except Exception as e:
            return [f"Error retrieving info: {str(e)}"]
