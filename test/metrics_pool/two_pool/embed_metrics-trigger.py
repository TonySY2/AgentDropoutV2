import json
import os
import numpy as np
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict, Any


INPUT_FILE = "deduped-mixed_metrics_two_pool.json"

OUTPUT_CACHE_FILE = "deduped-mixed_two_pool-trigger.jsonl"


EMBEDDING_MODEL = "###"
EMBEDDING_API_KEY = "EMPTY"
EMBEDDING_BASE_URL = "###"

CONCURRENCY_LIMIT = 50  #Embedded concurrency
# ==============================================================================

class MetricEmbedder:
    def __init__(self):

        self.embedding_client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
        self.existing_names = set()
        self._load_existing_cache()

    def _load_existing_cache(self):
  
        if os.path.exists(OUTPUT_CACHE_FILE):
            print(f"[Init] Loading existing cache from {OUTPUT_CACHE_FILE}...")
            count = 0
            try:
                with open(OUTPUT_CACHE_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            record = json.loads(line)
                            if "name" in record:
                                self.existing_names.add(record["name"])
                                count += 1
                        except:
                            pass
            except Exception as e:
                print(f"[Init Warning] Failed to load cache file: {e}")
            print(f"[Init] Found {count} existing embeddings. These will be skipped.")

    async def get_embedding(self, text: str) -> List[float]:
  
        if not text:
            return []
            
        async with self.semaphore:
            try:
                for _ in range(3): 
                    try:
                        response = await self.embedding_client.embeddings.create(
                            model=EMBEDDING_MODEL, 
                            input=[text]
                        )
                        embedding = np.array(response.data[0].embedding, dtype=np.float32)
                        
                
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                            
                        return embedding.tolist()
                    except Exception as e:
                        if "400" in str(e): # 
                            print(f"[Embed Error] Bad Request for text: {text[:50]}...")
                            return []
                        await asyncio.sleep(1)
                return []
            except Exception as e:
                print(f"\n[Fatal Error] Embedding API failed: {e}")
                return []

    async def process_item(self, metric: Dict[str, Any]):

        name = metric.get("name")
        
   
        if name in self.existing_names:
            return

     
        trigger = metric.get("evaluator_prompt", {}).get("trigger_condition", "")
        definition = metric.get("detailed_definition", "")
        
    
        if trigger and trigger != "N/A":
            text_to_embed = trigger  
        else:
           
            text_to_embed = definition

        if not text_to_embed:
            print(f"[Skip] Metric '{name}' has no content to embed.")
            return


        vector = await self.get_embedding(text_to_embed)
        
        if vector:
     
            record = {
                "name": name,
                "vector": vector,
                "embedded_text_snippet": text_to_embed[:100] 
            }
            try:
                with open(OUTPUT_CACHE_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"[Save Error] {e}")

    async def run(self):
    
        if not os.path.exists(INPUT_FILE):
            print(f"[Error] Input file not found: {INPUT_FILE}")
            return

  
        print(f"Loading metrics from {INPUT_FILE}...")
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                metrics_pool = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return

  
        to_process = [m for m in metrics_pool if m.get("name") not in self.existing_names]
        
        print(f"Total metrics: {len(metrics_pool)}")
        print(f"Already cached: {len(self.existing_names)}")
        print(f"Remaining to embed: {len(to_process)}")

        if not to_process:
            print("All metrics are already embedded. Done.")
            return

  
        tasks = [self.process_item(m) for m in to_process]
        await tqdm_asyncio.gather(*tasks, desc="Generating Embeddings")

        print(f"\nSuccess! Embeddings saved to: {OUTPUT_CACHE_FILE}")

if __name__ == "__main__":
    embedder = MetricEmbedder()
    asyncio.run(embedder.run())