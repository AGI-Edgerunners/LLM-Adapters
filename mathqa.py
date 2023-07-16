import os
import json
from datasets import load_dataset

dataset = load_dataset("math_qa")
save_path = "dataset/mathqa/test.json"

if not os.path.exists("dataset/mathqa/"):
    os.makedirs("dataset/mathqa/")


def writer(data, save_path):
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

test_data = []
for sample in dataset["test"]:
    options = sample["options"].replace("a", "A").replace("b", "B").replace("c", "C").replace("d", "D").replace("e", "E").replace("f", "F")
    test_data.append({
        "instruction": f"{sample['Problem']} The options: {options}",
        "input": "",
        "output": "",
        "answer": sample["correct"].upper(),
    })

writer(test_data, save_path)

