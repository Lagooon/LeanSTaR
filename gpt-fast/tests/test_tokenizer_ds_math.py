# tests with BPE tokenizer

from transformers import AutoTokenizer

ds_math_model_name = "deepseek-ai/deepseek-math-7b-base"
tokenizer = AutoTokenizer.from_pretrained(ds_math_model_name)

tests = [
    "",
    " ",
    "  ",
    "   ",
    "\t",
    "\n",
    "\t\n",
    "\n\nHello",
    "Hello Hello",
    "Hello world",
    " Hello world",
    "Hello World",
    " Hello World",
    " Hello World!",
    "Hello, world!",
    " Hello, world!",
    " this is 🦙.cpp",
    "w048 7tuijk dsdfhu",
    "нещо на Български",
    "កាន់តែពិសេសអាចខលចេញ",
    "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
    "Hello",
    " Hello",
    "  Hello",
    "   Hello",
    "    Hello",
    "    Hello\n    Hello",
    "\n =",
    "' era",
    "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
]

for text in tests:
    print("text: ", [text])
    print(tokenizer.encode(text))
    print([tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)])
