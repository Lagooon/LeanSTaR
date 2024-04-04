"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import time

import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict

import shortuuid
import torch
from tqdm import tqdm

from fastchat.conversation import Conversation, get_conv_template
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from vllm import LLM, SamplingParams

import torch
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel
from peft.tuners.lora import LoraLayer

torch.backends.cuda.matmul.allow_tf32 = True


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    ADD_NEW_LINE_DOUBLE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    DROMEDARY = auto()
    LLAMA2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_DOUBLE:
            ret = self.system + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i % 2 == 0:
                        ret += role + "\n" + message + "\n"
                    else:
                        ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.DROMEDARY:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += self.system + message
                    else:
                        ret += role + " " + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


conv_granite = Conversation(
    name="granite",
    system="",
    messages=(),
    roles=("<|user|>", "<|assistant|>"),
    sep_style=SeparatorStyle.ADD_NEW_LINE_DOUBLE,
    sep="</s>",
    offset=0,
)

conv_dromedary = Conversation(
    name="dromedary",
    system=(
        "# Dromedary\n\n"
        "## System Overview\n\n"
        "Consider an AI assistant whose codename is Dromedary, developed by the Self-Align team. "
        "Dromedary is trained on data up until Sept-2022, and it endeavors to be a helpful, ethical and reliable assistant.\n\n"
        "## User Conversation"
    ),
    roles=("User", "Dromedary"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DROMEDARY,
    sep="\n\n### ",
    stop_str="### User",
)


def run_eval(
    model_path,
    model_prompt,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    temperature,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    # use_ray = num_gpus_total > 1
    use_ray = False

    if use_ray:
        # get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
        #     get_model_answers
        # ).remote
        pass
    else:
        get_answers_func = get_model_answers

    local_rank = int(os.getenv("RANK", 0))
    print("local_rank", local_rank)
    chunk_size = ((len(questions) - 1) // (num_gpus_total // num_gpus_per_model)) + 1
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        if i // chunk_size == local_rank:
            ans_handles.append(
                get_answers_func(
                    model_path,
                    model_prompt,
                    model_id,
                    questions[i : i + chunk_size],
                    answer_file,
                    max_new_token,
                    num_choices,
                    temperature,
                    num_gpus_per_model,
                    max_gpu_memory,
                )
            )

    if use_ray:
        # ray.get(ans_handles)
        pass


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_prompt,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    temperature,
    num_gpus_per_model,
    max_gpu_memory,
):
    model = LLM(model=model_path, tensor_parallel_size=num_gpus_per_model)

    generated_question_ids = []
    if os.path.exists(os.path.expanduser(answer_file)):
        with open(os.path.expanduser(answer_file), "r") as f:
            for line in f:
                try:
                    generated_question_ids.append(json.loads(line)["question_id"])
                except json.decoder.JSONDecodeError:
                    print("JSONDecodeError")
    generated_question_ids = set(generated_question_ids)

    batch_size = num_gpus_per_model * 64 // num_choices

    batched_prompts = []

    top_p = 1.0
    sampling_params = SamplingParams(
        n=num_choices,
        max_tokens=max_new_token,
        temperature=temperature,
        top_p=top_p,
    )

    batched_outputs = []
    seed = 0

    for question in tqdm(questions):
        if question["question_id"] in generated_question_ids:
            continue

        seed += 1
        torch.manual_seed(seed)
        if model_prompt == "dromedary":
            conv = conv_dromedary.copy()
        else:
            conv = get_conversation_template(model_prompt or model_id)
            print(conv)

        assert len(question["turns"]) == 1

        qs = question["turns"][0]
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        batched_prompts.append((question["question_id"], prompt))

        if len(batched_prompts) == batch_size:
            outputs = model.generate([x[-1] for x in batched_prompts], sampling_params)
            new_outputs = []
            for output in outputs:
                per_question_outputs = []
                for i, sub_output in enumerate(output.outputs):
                    per_question_outputs.append((i, sub_output.text))
                new_outputs.append(per_question_outputs)
            outputs = new_outputs

            for (question_id, prompt), per_question_outputs in zip(
                batched_prompts, outputs
            ):
                for i, output in per_question_outputs:
                    if conv.stop_str:
                        output = output[: output.find(conv.stop_str)]
                    output = output.strip()
                    batched_outputs.append(
                        (question_id, {"index": i, "turns": [output]})
                    )

            batched_prompts = []

    if len(batched_prompts) > 0:
        outputs = model.generate([x[-1] for x in batched_prompts], sampling_params)
        new_outputs = []
        for output in outputs:
            per_question_outputs = []
            for i, sub_output in enumerate(output.outputs):
                per_question_outputs.append((i, sub_output.text))
            new_outputs.append(per_question_outputs)
        outputs = new_outputs

        for (question_id, prompt), per_question_outputs in zip(
            batched_prompts, outputs
        ):
            for i, output in per_question_outputs:
                if conv.stop_str:
                    output = output[: output.find(conv.stop_str)]
                output = output.strip()
                batched_outputs.append((question_id, {"index": i, "turns": [output]}))

    # merge outputs by question id
    question_id_to_outputs = {}
    for output in batched_outputs:
        question_id, output = output
        if question_id not in question_id_to_outputs:
            question_id_to_outputs[question_id] = []
        question_id_to_outputs[question_id].append(output)

    # Dump answers
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    print("total number of questions:", len(question_id_to_outputs))
    import fcntl

    with open("/tmp/file_lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_UN)
        with open(os.path.expanduser(answer_file), "a") as fout:
            output_str = ""
            for question_id, outputs in question_id_to_outputs.items():
                ans_json = {
                    "question_id": question_id,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": outputs,
                    "tstamp": time.time(),
                }
                output_str += json.dumps(ans_json) + "\n"
            fout.write(output_str)
        fcntl.flock(f, fcntl.LOCK_UN)


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    local_rank = int(os.getenv("RANK", 0))

    if local_rank != 0:
        print("local_rank:", [local_rank], "skip reorg_answer_file")
        return

    answer_file = os.path.expanduser(answer_file)
    print("Reorg:", answer_file)

    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            try:
                qid = json.loads(l)["question_id"]
                answers[qid] = l
            except json.decoder.JSONDecodeError:
                print("JSONDecodeError")

    qids = sorted(list(answers.keys()))
    print(f"Total number of questions: {len(qids)}")
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-prompt",
        type=str,
        default=None,
        help="The prompt to use for the model.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    args = parser.parse_args()

    # if args.num_gpus_total > 1:
    #     import ray

    #     ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_prompt,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.temperature,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
    )

    # reorg_answer_file(answer_file)
