import json

## From
# {
#     "question": "We roll a fair 6-sided die 5 times.  What is the probability that we get a 6 in at most 2 of the rolls?",
#     "answer": "\\frac{625}{648}",
#     "answer_detail": "The number of ways to roll exactly 2 6's is $\\binom{5}{2}5^3$, since there are $\\binom{5}{2}$ choices for which of the two dice are 6, and there are 5 choices for each of the other 3 dice. Similarly, the number of ways to roll exactly 1 6 is $\\binom{5}{1}5^4$, and the number of ways to roll no 6's is $\\binom{5}{0}5^5$. So the probability is \\[\\frac{\\binom{5}{2}5^3+\\binom{5}{1}5^4+\\binom{5}{0}5^5}{6^5}=\\boxed{\\frac{625}{648}}.\\]",
#     "level": "Level 5",
#     "type": "Counting & Probability",
#     "question_id": "396"
# },
## To
# {"question_id": 81, "category": "writing", "turns": ["[instruction]\n\n[question]"]}

INSTRUCTION = (
    "I will give you a mathematical problem, either pure math or a word problem. "
    'Your response should always start with the phrase "Let\'s think step by step." '
    "Following this, you should provide a clear and logical breakdown of the problem, detailing each step and any calculations or reasonings involved. "
    "This should be written in a way that even someone unfamiliar with the problem can understand. "
    'Conclude your response with the phrase "The answer is: [ANSWER]", where "[ANSWER]" is the final solution to the problem. '
    "Any mathematical symbols, equations, or expressions should be accurately represented and clear to understand."
)

if __name__ == "__main__":
    input_file = "/workspace/zhiqings/output3/data/MATH_test-cleaned.json"
    output_file = "data/math_debug/question.jsonl"

    with open(input_file, "r") as f:
        data = json.load(f)

    # 1. we check if question_id is unique and its max and min values

    question_ids = []

    for d in data:
        question_ids.append(int(d["question_id"]))

    print("max question_id: ", max(question_ids))
    print("min question_id: ", min(question_ids))
    print("len question_ids: ", len(question_ids))
    print("len set question_ids: ", len(set(question_ids)))

    # 2. it seems the question_ids are not unique, we will assign new ids to each question

    output_data = []

    for i, d in enumerate(data[:128]):
        output_data.append(
            {
                "question_id": i + 1,
                "category": "math",
                "turns": [INSTRUCTION + "\n\n" + d["question"]],
                "short_answer": d["answer"],
                "level": d["level"],
                "type": d["type"],
            }
        )

    # 3. we save the data

    with open(output_file, "w") as f:
        for d in output_data:
            f.write(json.dumps(d) + "\n")
    print("Saved to: ", output_file)
