import json
import argparse
import glob
import os
import sys
import math

def get_task_id_safe(instance: dict, default_idx: int) -> str:


    if 'unique_id' in instance:
        return str(instance['unique_id']).strip()
        

    if 'id' in instance:
        return str(instance['id']).strip()
        

    for key in ['task_id', 'question_id', 'problem_id', 'name']:
        if key in instance and instance[key] is not None:
            return str(instance[key]).strip()
            

    return f"auto_gen_id_{default_idx}"

def load_input_data(file_path):
 
    if not os.path.exists(file_path):
        print(f"[FATAL]: {file_path}")
        sys.exit(1)

    print(f"  -> loading: {file_path}")
    raw_items = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
           
            try:
                content = f.read().strip()
                if not content: return []
                
                parsed = json.loads(content)
                
            
                if isinstance(parsed, dict):
                    if "test" in parsed and isinstance(parsed["test"], list):
                        print("    [Info](key='test')")
                        return parsed["test"]
                    elif "data" in parsed and isinstance(parsed["data"], list):
                        print("    [Info]  (key='data')")
                        return parsed["data"]
                    else:
                     
                        print("    [Info] ")
                        return [parsed]
                
         
                elif isinstance(parsed, list):
                    print("    [Info] ")
                    return parsed
                
            except json.JSONDecodeError:
               
                print("  ...")
                f.seek(0)
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    try:
                        raw_items.append(json.loads(line))
                    except json.JSONDecodeError:
                     
                        pass
                return raw_items
                
    except Exception as e:
        print(f"[FATAL] : {e}")
        sys.exit(1)
        
    return raw_items

def get_finished_ids(out_dir):
   
    finished = set()
    if not os.path.exists(out_dir):
        print(f"  ->  {out_dir} no found")
        return finished

    
    output_files = glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True)
    
    if not output_files:
        return finished
        
    print(f"  -> scaning {len(output_files)} ...")
    
    for file_path in output_files:
    
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
         
                if isinstance(data, dict):
                    finished.update(str(k) for k in data.keys())
                
         
                elif isinstance(data, list):
                    for item in data:
                 
                        fid = str(item.get('id') or item.get('task_id') or item.get('question_id') or '').strip()
                        if fid:
                            finished.add(fid)
        except:
            pass
            
    print(f"found   {len(finished)}")
    return finished

def main():
    parser = argparse.ArgumentParser(description="Universal Task Distributor")
    parser.add_argument('--in_file', type=str, required=True, help="")
    parser.add_argument('--out_dir', type=str, required=True, help="")
    parser.add_argument('--node_num', type=int, required=True, help="")
    parser.add_argument('--tmp_dir', type=str, default='./tmp', help="")
    args = parser.parse_args()

    print(f"========== Universal Task Distributor ==========")
    print(f"Input:  {args.in_file}")
    print(f"Output: {args.out_dir}")
    print(f"Nodes:  {args.node_num}")
    

    os.makedirs(args.tmp_dir, exist_ok=True)
    
  
    raw_data = load_input_data(args.in_file)
    if not raw_data:
        print("[ERROR]")
        sys.exit(1)

  
    normalized_tasks = []
   
    seen_ids = set()
    
    for idx, item in enumerate(raw_data):
    
        tid = get_task_id_safe(item, idx)
        
        
        item['id'] = tid
        
     
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        
        normalized_tasks.append(item)
    
    
    try:
        normalized_tasks.sort(key=lambda x: int(x['id']))
    except:
        normalized_tasks.sort(key=lambda x: str(x['id']))
        
    print(f"  ->  {len(normalized_tasks)}")


    finished_ids = get_finished_ids(args.out_dir)
    pending_tasks = [t for t in normalized_tasks if t['id'] not in finished_ids]
    
    print(f"  ->: {len(pending_tasks)}")


    if len(pending_tasks) == 0:
 
        for i in range(args.node_num):
            with open(os.path.join(args.tmp_dir, f"part_{i}.jsonl"), 'w') as f: pass
    else:
      
        chunk_size = math.ceil(len(pending_tasks) / args.node_num)
        

        
        for i in range(args.node_num):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(pending_tasks))
            
            chunk = []
            if start < len(pending_tasks):
                chunk = pending_tasks[start:end]
            
            out_file = os.path.join(args.tmp_dir, f"part_{i}.jsonl")
            
            with open(out_file, 'w', encoding='utf-8') as f:
                for item in chunk:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # print(f"    - Node {i}: {len(chunk)} tasks -> {out_file}")

    print("========== Distribution Complete ==========\n")

if __name__ == '__main__':
    main()