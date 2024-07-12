import json
data = []
with open(f"data/leandojo_benchmark_4/processed/STaR-generated-train.json", "r") as f:
    data = data + json.load(f)
with open(f"data/leandojo_benchmark_4/processed/gpt4-generated-train-1.json", "r") as f:
    data = data + json.load(f)
print(len(data))
with open(f"data/leandojo_benchmark_4/processed/STaR-generated-train-1.json", "w") as f:
    json.dump(data, f)
