import json
import argparse
import glob
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--node_num', type=int, required=True)
    parser.add_argument('--tmp_dir', type=str, default='./tmp')
    args = parser.parse_args()

    print(f"--- [get_tasks.py] ---")
    print(f"in: {args.in_file}")
    
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.in_file):
        print(f"[FATAL ERROR] no found: {os.path.abspath(args.in_file)}")
        sys.exit(1)

    input_data = []

    try:
        with open(args.in_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
          
                print("[INFO]...")
                input_data = json.load(f)
            else:
          
                print("[INFO] ...")
                for line in f:
                    line = line.strip()
                    if not line: continue
                    input_data.append(json.loads(line))
        

        for idx, item in enumerate(input_data):
            if 'id' not in item and 'idx' not in item:
                item['id'] = str(idx) 
            elif 'id' in item:
                item['id'] = str(item['id'])
            elif 'idx' in item: 
                item['id'] = str(item['idx'])
                
        print(f"successfully load {len(input_data)} ")
        
    except json.JSONDecodeError as e:
        print(f"[FATAL ERROR] JSON : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL ERROR] : {e}")
        sys.exit(1)

    finished_ids = set()
    output_files = glob.glob(f"{args.out_dir}/part_*.json")
    
    for out_file_path in output_files:
        if os.path.getsize(out_file_path) == 0: continue
        try:
            with open(out_file_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
                if isinstance(output_data, dict):
            
                    finished_ids.update(output_data.keys())
        except Exception:
            pass 


    unfinished_tasks = []
    for instance in input_data:

        task_id = str(instance.get('id', ''))
        if task_id and task_id not in finished_ids:
            unfinished_tasks.append(instance)

    print(f"all: {len(input_data)} | over: {len(finished_ids)} | waiting: {len(unfinished_tasks)}")


    if not unfinished_tasks:

        for node_id in range(args.node_num):
            open(os.path.join(args.tmp_dir, f"part_{node_id}.jsonl"), 'w').close()
        return

    chunk_size = (len(unfinished_tasks) + args.node_num - 1) // args.node_num
    
    for node_id in range(args.node_num):
        start = node_id * chunk_size
        end = (node_id + 1) * chunk_size
        chunk_data = unfinished_tasks[start:end] if chunk_size > 0 else []
        
        out_file = os.path.join(args.tmp_dir, f"part_{node_id}.jsonl")
        
        with open(out_file, 'w', encoding='utf-8') as f:
            for instance in chunk_data:
                f.write(json.dumps(instance, ensure_ascii=False) + '\n')
        
    print(f" {args.node_num}  /to {args.tmp_dir}")

if __name__ == '__main__':
    main()