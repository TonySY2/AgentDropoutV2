

import json
import os
import numpy as np
import asyncio
import argparse
from openai import AsyncOpenAI
from json_repair import repair_json
from typing import List, Dict, Any




class PipelineProcessor:
    def __init__(self, args):
        self.input_data_file = args.input_data_file
        self.output_metrics_file = args.output_metrics_file
        self.output_embedding_file = args.output_embedding_file
        self.llm_model = args.llm_model
        self.embedding_model = args.embedding_model
        self.llm_client = AsyncOpenAI(api_key=args.llm_key, base_url=args.llm_url)
        self.embedding_client = AsyncOpenAI(api_key=args.embedding_key, base_url=args.embedding_url)
        
  
        self.unique_pool = []
        self.existing_names = set()

    async def get_embedding(self, text: str) -> np.ndarray:
   
        if not text: return np.array([])
        try:
            for _ in range(3):
                try:
                    response = await self.embedding_client.embeddings.create(
                        model=self.embedding_model,
                        input=[text]
                    )
                    vec = np.array(response.data[0].embedding, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0: vec = vec / norm
                    return vec
                except Exception:
                    await asyncio.sleep(1)
            return np.array([])
        except Exception as e:
            print(f"[Embed Error] {e}")
            return np.array([])

    def extract_raw_metrics(self, file_path: str) -> List[Dict]:
     
        print(f"Loading raw data from {file_path}...")
        if not os.path.exists(file_path):
            print("File not found!")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"JSON Load Error: {e}")
            return []

        raw_list = []
        seen_names_raw = set()
        
        for task_id, task_data in data.items():
            trajectories = task_data.get("trajectories", {})
            if not trajectories: continue
            for agent_name, steps in trajectories.items():
                if not isinstance(steps, list): continue
                for step in steps:
                    metrics = step.get("generated_adversarial_metrics", [])
                    if isinstance(metrics, list):
                        for m in metrics:
                            if "name" in m and "detailed_definition" in m:
                        
                                if m["name"] not in seen_names_raw:
                             
                                    if "metadata" not in m:
                                        m["metadata"] = {"source_task": task_id}
                                    raw_list.append(m)
                                    seen_names_raw.add(m["name"])
        print(f"Extracted {len(raw_list)} raw unique-named metrics.")
        return raw_list

    async def check_duplicate_llm(self, new_metric: dict, def_embedding: np.ndarray) -> bool:
     
        if not self.unique_pool:
            return False

  
        pool_vectors = np.stack([m['def_embedding'] for m in self.unique_pool])
        similarities = np.dot(pool_vectors, def_embedding)
        
        k = min(15, len(self.unique_pool))
        top_indices = np.argsort(similarities)[-k:][::-1]
        

        if similarities[top_indices[0]] < 0.65:
            return False

   
        candidates_text = ""
        for i, idx in enumerate(top_indices):
            m = self.unique_pool[idx]['data']
            sim = similarities[idx]
       
            candidates_text += (
                f"Candidate {i+1} (Sim: {sim:.2f}):\n"
                f"  - Name: {m['name']}\n"
                f"  - Definition: {m['detailed_definition']}\n\n"
            )

        prompt = f"""
Check if the NEW metric describes the SAME Logic Error as any candidate.
Focus on the **Detailed Definition**.

### New Metric
- Name: {new_metric['name']}
- Definition: {new_metric['detailed_definition']}

### Candidates (Retrieved by definition similarity)
{candidates_text}

Output JSON ONLY: {{"is_duplicate": true/false}}
"""
    
        try:
            resp = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            res = repair_json(resp.choices[0].message.content, return_objects=True)
            if isinstance(res, list): res = res[0]
            return res.get("is_duplicate", False)
        except:
            return False

    async def phase1_deduplication(self, raw_metrics: List[Dict]):
     
        print(f"\n>>> Phase 1: Deduplication (Based on Definition) <<<")
        
        for idx, m in enumerate(raw_metrics, start=1):
            print(f"[Dedup {idx}/{len(raw_metrics)}] {m.get('name', 'unknown')}")
            name = m.get("name")
            definition = m.get("detailed_definition", "")
            
            if not definition: continue

     
            def_emb = await self.get_embedding(definition)
            if def_emb.size == 0: continue

     
            is_dup = await self.check_duplicate_llm(m, def_emb)
            
            if not is_dup:
                self.unique_pool.append({
                    "data": m,
                    "def_embedding": def_emb
                })
                self.existing_names.add(name)
           
            else:
              
                pass
        
        print(f"Phase 1 Complete. Unique Metrics: {len(self.unique_pool)}")

    async def phase2_finalize(self):
   
        print(f"\n>>> Phase 2: Generating Trigger Embeddings & Saving <<<")
        
        final_metrics_list = []
        tasks = []


        for item in self.unique_pool:
            metric = item['data']
            final_metrics_list.append(metric) 
            
   
            trigger = metric.get("evaluator_prompt", {}).get("trigger_condition", "")
            definition = metric.get("detailed_definition", "")
            

            text_to_embed = trigger if (trigger and trigger != "N/A") else definition
            
            tasks.append(self._embed_trigger_and_format(metric["name"], text_to_embed))


        results = []
        for idx, task in enumerate(tasks, start=1):
            print(f"[Embed {idx}/{len(tasks)}]")
            results.append(await task)
        
   
        with open(self.output_metrics_file, "w", encoding="utf-8") as f:
            json.dump(final_metrics_list, f, indent=4, ensure_ascii=False)
            
 
        with open(self.output_embedding_file, "w", encoding="utf-8") as f:
            valid_count = 0
            for res in results:
                if res:
                    f.write(json.dumps(res) + "\n")
                    valid_count += 1
                    
        print(f"\nAll Done!")
        print(f"1. Clean Metrics saved to: {self.output_metrics_file}")
        print(f"2. Trigger Embeddings saved to: {self.output_embedding_file} (Count: {valid_count})")

    async def _embed_trigger_and_format(self, name: str, text: str) -> Dict:
  
        vec = await self.get_embedding(text)
        if vec.size > 0:
            return {
                "name": name,
                "vector": vec.tolist(),
                "trigger_text_preview": text[:100]
            }
        return {}

    async def run(self):

        raw_metrics = self.extract_raw_metrics(self.input_data_file)
        if not raw_metrics: return

  
        if os.path.exists(self.output_metrics_file): os.remove(self.output_metrics_file)
        if os.path.exists(self.output_embedding_file): os.remove(self.output_embedding_file)

     
        await self.phase1_deduplication(raw_metrics)

   
        await self.phase2_finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", default="results/part_0.json")
    parser.add_argument("--output_metrics_file", default="metrics_pool_deduped.json")
    parser.add_argument("--output_embedding_file", default="metrics_embeddings_trigger.jsonl")
    parser.add_argument("--llm_model", default=os.environ.get("SUPERVISOR_MODEL"))
    parser.add_argument("--llm_url", default=os.environ.get("SUPERVISOR_URL"))
    parser.add_argument("--llm_key", default=os.environ.get("SUPERVISOR_KEY", "EMPTY"))
    parser.add_argument("--embedding_model", default=os.environ.get("EMBEDDING_MODEL"))
    parser.add_argument("--embedding_url", default=os.environ.get("EMBEDDING_URL"))
    parser.add_argument("--embedding_key", default=os.environ.get("EMBEDDING_KEY", "EMPTY"))
    args = parser.parse_args()

    missing = [
        name for name in ("llm_model", "llm_url", "embedding_model", "embedding_url")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    pipeline = PipelineProcessor(args)
    asyncio.run(pipeline.run())
