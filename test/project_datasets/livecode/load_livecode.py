from datasets import load_dataset


ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v1", trust_remote_code=True)


ds['test'].to_json("livecodebench_v1.jsonl")

print("over, livecodebench_v1.jsonl")