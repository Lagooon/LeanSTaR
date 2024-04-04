import os
import json
import glob


def print_table(results_table, type_list, level_list):
    # Header for levels
    print(f"{'Type/Level':<24}", end="")
    for level in level_list:
        print(f"{level:>16}", end="")
    print(f"{'Overall':>16}")

    # Rows for types
    for type_name in type_list:
        print(f"{type_name:<24}", end="")
        for level_name in level_list:
            correct, total = results_table[type_name].get(level_name, [0, 0])
            accuracy = correct / total if total > 0 else 0
            print(f"{accuracy:>16.2f}", end="")
        # Overall accuracy for type
        correct, total = results_table[type_name].get("Overall", [0, 0])
        accuracy = correct / total if total > 0 else 0
        print(f"{accuracy:>16.2f}")
    print()

    # Overall row at the bottom
    print(f"{'Overall':<24}", end="")
    for level_name in level_list:
        correct, total = results_table["Overall"].get(level_name, [0, 0])
        accuracy = correct / total if total > 0 else 0
        print(f"{accuracy:>16.2f}", end="")
    # Overall accuracy for all types and levels
    correct, total = results_table["Overall"]["Overall"]
    accuracy = correct / total if total > 0 else 0
    print(f"{accuracy:>16.2f}")


if __name__ == "__main__":
    gt_path = "data/math/question.jsonl"
    model_answer_dir = "data/math/model_answer"

    with open(gt_path, "r") as f:
        gt = [json.loads(line) for line in f]

    for file in glob.glob(os.path.join(model_answer_dir, "*scored*.jsonl")):
        results_table = {}
        type_list = set()
        level_list = set()
        file = file.split("/")[-1]
        model_answer_path = os.path.join(model_answer_dir, file)
        with open(model_answer_path, "r") as f:
            model_answer = [json.loads(line) for line in f]

        # sort model_answer with question_id
        model_answer = sorted(model_answer, key=lambda x: x["question_id"])

        assert len(gt) == len(model_answer)

        for i in range(len(gt)):
            gt_answer = gt[i]["short_answer"]
            if 'Level' in gt[i]['level']:
                level = f"Level {gt[i]['level'][-1]}"
            else:
                level = f"Level 0"
            question_type = gt[i]["type"]
            if "turns" in model_answer[i]["choices"][0]:
                keyword = "turns"
            else:
                keyword = "output"

            model_short_answer = "I don't know"
            model_short_answer_score = -1e3

            for answer in model_answer[i]["choices"]:
                answer_candidate = answer[keyword][0].split("\nThe answer is:")[-1].strip()
                answer_score = answer["score"]

                if answer_score > model_short_answer_score:
                    model_short_answer = answer_candidate
                    model_short_answer_score = answer_score

            # Add to sets for later use
            type_list.add(question_type)
            level_list.add(level)

            # Initialize dictionary entries if they do not exist
            if question_type not in results_table:
                results_table[question_type] = {}
            if level not in results_table[question_type]:
                results_table[question_type][level] = [0, 0]

            # Update counts
            if gt_answer == model_short_answer:
                results_table[question_type][level][0] += 1
                if "Overall" not in results_table[question_type]:
                    results_table[question_type]["Overall"] = [0, 0]
                results_table[question_type]["Overall"][0] += 1
            results_table[question_type][level][1] += 1
            if "Overall" not in results_table[question_type]:
                results_table[question_type]["Overall"] = [0, 0]
            results_table[question_type]["Overall"][1] += 1

            # Overall accuracy
            if "Overall" not in results_table:
                results_table["Overall"] = {}
            if level not in results_table["Overall"]:
                results_table["Overall"][level] = [0, 0]
            results_table["Overall"][level][1] += 1
            if gt_answer == model_short_answer:
                results_table["Overall"][level][0] += 1
            if "Overall" not in results_table["Overall"]:
                results_table["Overall"]["Overall"] = [0, 0]
            results_table["Overall"]["Overall"][1] += 1
            if gt_answer == model_short_answer:
                results_table["Overall"]["Overall"][0] += 1

        print(file)
        # Sort the lists
        type_list = sorted(type_list)
        level_list = sorted(level_list, key=lambda x: int(x[-1]))

        # Print the table
        print_table(results_table, type_list, level_list)
        print()
