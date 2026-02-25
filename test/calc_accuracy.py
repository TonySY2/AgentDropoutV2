import json
import os
import re


FILE_PATH = "###-part_0.json"  #The path of the test result file
# ===========================================

def load_messy_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    items = []
    

    try:
        data = json.loads(content)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        return items
    except json.JSONDecodeError:
        pass


    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        
        try:
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            
  
            if isinstance(obj, dict):
                first_value = next(iter(obj.values())) if obj else None
                if len(obj) == 1 and isinstance(first_value, dict) and ('answer' in first_value or 'hypothesis' in first_value or 'is_correct' in first_value):
                    items.append(first_value)
                else:
                    items.append(obj)
            
            pos = end_pos
        except json.JSONDecodeError:
            pos += 1
            
    return items

def check_entry_correctness(entry, index):
    bool_keys = ['is_correct', 'solved', 'is_solved', 'is_correct_prediction']
    for key in bool_keys:
        if key in entry:
            val = entry[key]
            if isinstance(val, str):
                return val.lower() == 'true'
            return bool(val)

    if 'answer' in entry and 'hypothesis' in entry:
        ans = str(entry['answer']).strip()
        hyp = str(entry['hypothesis']).strip()

        return ans.lower() == hyp.lower()


    if 'gold_value' in entry and 'hypothesis' in entry:
        ans = str(entry['gold_value']).strip()
        hyp = str(entry['hypothesis']).strip()
        return ans.lower() == hyp.lower()


    print(f"[Warning] Item {index} has no verification keys. Keys found: {list(entry.keys())}")
    return False

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    print(f"Loading data from: {FILE_PATH} ...")
    data_items = load_messy_json(FILE_PATH)
    
    total = len(data_items)
    correct_count = 0
    
    print(f"Found {total} entries. Calculating accuracy...\n")
    
    for i, item in enumerate(data_items):

        if len(item) == 1 and isinstance(list(item.values())[0], dict):
            real_item = list(item.values())[0]
        else:
            real_item = item
            
        is_right = check_entry_correctness(real_item, i)
        
        if is_right:
            correct_count += 1
        


    if total == 0:
        print("No valid data found.")
        return

    accuracy = (correct_count / total) * 100
    
    print("-" * 30)
    print(f"Total Tasks : {total}")
    print(f"Correct     : {correct_count}")
    print(f"Incorrect   : {total - correct_count}")
    print("-" * 30)
    print(f"Accuracy    : {accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()