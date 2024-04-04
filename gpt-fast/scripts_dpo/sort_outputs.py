import json
import fire

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def main(
    input_file: str,
    output_file: str,
    model_name: str,
    original_input_file: str,
):
    with open(original_input_file, "r") as f:
        original_inputs = json.load(f)

    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    data = sorted(
        data,
        key=lambda x: x["idx"],
    )

    # remove duplicated idx
    filtered_data = []
    last_idx = -1
    for d in data:
        if d["idx"] != last_idx:
            filtered_data.append(
                {
                    "output": d["output"],
                    "generator": model_name,
                }
            )
            last_idx = d["idx"]

    assert len(original_inputs) == len(filtered_data)

    max_tokenized_output_length = -1

    for i in range(len(original_inputs)):
        filtered_data[i]["instruction"] = (
            original_inputs[i]["instruction"] + "\n\n" + original_inputs[i]["input"]
            if len(original_inputs[i]["input"]) > 0
            else original_inputs[i]["instruction"]
        )
        filtered_data[i]["dataset"] = original_inputs[i]["dataset"]
        filtered_data[i]["datasplit"] = original_inputs[i]["datasplit"]

        tokenized_output = tokenizer(filtered_data[i]["output"])["input_ids"]
        max_tokenized_output_length = max(
            max_tokenized_output_length, len(tokenized_output)
        )

    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Saved to {output_file}")
    print(f"Max tokenized output length: {max_tokenized_output_length}")


if __name__ == "__main__":
    fire.Fire(main)
