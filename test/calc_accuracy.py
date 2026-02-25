import json
import os
import re

# ================= 配置区域 =================
# 在这里填写你的结果文件路径 (支持绝对路径或相对路径)
FILE_PATH = "results_main_table/results-aqua/autogen-base/part_0.json" 
# ===========================================

def load_messy_json(filepath):
    """
    尝试加载各种奇怪格式的 JSON 文件：
    1. 标准 JSON
    2. 堆叠的 JSON 对象 (Concatenated JSON)
    3. 包含单个大字典的 JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    items = []
    
    # 尝试 1: 标准 JSON 加载
    try:
        data = json.loads(content)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # 如果是一个大字典，values 通常是我们需要的数据
            # 例如: {"0": {"answer":...}, "1": {"answer":...}}
            items = list(data.values())
        return items
    except json.JSONDecodeError:
        pass

    # 尝试 2: 堆叠 JSON (即文件里有多个 {} {} {})
    # 使用 raw_decode 逐个解析
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # 跳过空白字符
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        
        try:
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            
            # 处理解析出的对象
            if isinstance(obj, dict):
                # 检查这个 dict 是否是 wrapper (例如 key 是 ID, value 是数据)
                # 你的数据很多是 {"0": {...}} 这种形式
                first_value = next(iter(obj.values())) if obj else None
                if len(obj) == 1 and isinstance(first_value, dict) and ('answer' in first_value or 'hypothesis' in first_value or 'is_correct' in first_value):
                    items.append(first_value)
                else:
                    items.append(obj)
            
            pos = end_pos
        except json.JSONDecodeError:
            # 如果解析失败，尝试跳过当前字符继续找下一个可能的 {
            pos += 1
            
    return items

def check_entry_correctness(entry, index):
    """
    核心判分逻辑，适应多种 Key
    """
    # 1. 优先检查明确的布尔值标记
    bool_keys = ['is_correct', 'solved', 'is_solved', 'is_correct_prediction']
    for key in bool_keys:
        if key in entry:
            val = entry[key]
            # 处理字符串形式的 "true"/"false"
            if isinstance(val, str):
                return val.lower() == 'true'
            return bool(val)

    # 2. 如果没有标记，进行 Answer vs Hypothesis 对比
    # 你的数据里 AQuA 只有 answer 和 hypothesis
    if 'answer' in entry and 'hypothesis' in entry:
        ans = str(entry['answer']).strip()
        hyp = str(entry['hypothesis']).strip()
        
        # 简单的相等判断 (忽略大小写)
        # 比如 AQuA 的 "A" == "A"
        # 比如 Math 的 "70" == "70"
        return ans.lower() == hyp.lower()

    # 3. 兜底：如果有些只有 gold_value
    if 'gold_value' in entry and 'hypothesis' in entry:
        ans = str(entry['gold_value']).strip()
        hyp = str(entry['hypothesis']).strip()
        return ans.lower() == hyp.lower()

    # 如果所有判断条件都不具备，视为 Unverifiable (通常算错，或者打印警告)
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
        # 容错：有些 item 可能是空的或者结构不对，再次尝试提取内部值
        # 针对 {"HumanEval_0...": {...}} 这种结构，如果在 load 阶段没解开
        if len(item) == 1 and isinstance(list(item.values())[0], dict):
            real_item = list(item.values())[0]
        else:
            real_item = item
            
        is_right = check_entry_correctness(real_item, i)
        
        if is_right:
            correct_count += 1
        
        # 可选：打印每个任务的状态
        # status_icon = "✅" if is_right else "❌"
        # print(f"{status_icon} Item {i}: {real_item.get('id', 'NoID')}")

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