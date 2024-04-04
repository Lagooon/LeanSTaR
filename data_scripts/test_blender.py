import llm_blender

blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")

inputs = ["hello, how are you!", "I love you!"]
candidates_texts = [
    ["get out!", "hi! I am fine, thanks!", "bye!"],
    ["I love you too!", "I hate you!", "Thanks! You're a good guy!"],
]

scores = blender.rank(
    inputs, candidates_texts, return_scores=True, batch_size=2, policy="max_probs"
)
scores = (scores * 3 - 0.5) / 2.0

print(scores, scores.sum(axis=1))
