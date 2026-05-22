import json
import os
import numpy as np
import asyncio
import argparse
from openai import AsyncOpenAI
from typing import List, Dict, Any


# ==============================================================================

class MetricEmbedder:
    def __init__(self, input_file: str, output_cache_file: str, embedding_model: str, embedding_base_url: str, embedding_api_key: str):

        self.input_file = input_file
        self.output_cache_file = output_cache_file
        self.embedding_model = embedding_model
        self.embedding_client = AsyncOpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
    
        self.existing_names = set()
        self._load_existing_cache()

    def _load_existing_cache(self):
  
        if os.path.exists(self.output_cache_file):
            print(f"[Init] Loading existing cache from {self.output_cache_file}...")
            count = 0
            try:
                with open(self.output_cache_file, 'r', encoding='utf-8') as f:
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
            
        try:
            for _ in range(3):
                try:
                    response = await self.embedding_client.embeddings.create(
                        model=self.embedding_model,
                        input=[text]
                    )
                    embedding = np.array(response.data[0].embedding, dtype=np.float32)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    return embedding.tolist()
                except Exception as e:
                    if "400" in str(e):
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
                with open(self.output_cache_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"[Save Error] {e}")

    async def run(self):
    
        if not os.path.exists(self.input_file):
            print(f"[Error] Input file not found: {self.input_file}")
            return

  
        print(f"Loading metrics from {self.input_file}...")
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
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

  
        for idx, metric in enumerate(to_process, start=1):
            print(f"[{idx}/{len(to_process)}] {metric.get('name', 'unknown')}")
            await self.process_item(metric)

        print(f"\nSuccess! Embeddings saved to: {self.output_cache_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="deduped-mixed_metrics_two_pool.json")
    parser.add_argument("--output_cache_file", default="deduped-mixed_two_pool-trigger.jsonl")
    parser.add_argument("--embedding_model", default=os.environ.get("EMBEDDING_MODEL"))
    parser.add_argument("--embedding_url", default=os.environ.get("EMBEDDING_URL"))
    parser.add_argument("--embedding_key", default=os.environ.get("EMBEDDING_KEY", "EMPTY"))
    args = parser.parse_args()

    if not args.embedding_model:
        raise SystemExit("Set EMBEDDING_MODEL or pass --embedding_model.")
    if not args.embedding_url:
        raise SystemExit("Set EMBEDDING_URL or pass --embedding_url.")

    embedder = MetricEmbedder(
        input_file=args.input_file,
        output_cache_file=args.output_cache_file,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_url,
        embedding_api_key=args.embedding_key,
    )
    asyncio.run(embedder.run())
