from pathlib import Path
from sentencepiece import SentencePieceProcessor
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from data_utils.tokenizer_utils import FakePreTrainedTokenizer

if __name__ == "__main__":
    checkpoint_path = Path(
        "/nobackup/users/yikangs/zhiqings/math/checkpoints/EleutherAI/llemma_34b/model.pth"
    )

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

    print("eos_id", tokenizer.eos_id())
    print("unk_id", tokenizer.unk_id())
    print("pad_id", tokenizer.pad_id())
    print("bos_id", tokenizer.bos_id())

    print(tokenizer.encode("1+1"))
    print(tokenizer.encode(["1+1", "2+2+2+2"]))

    for text in [
        "\nI think",
        "\n\nI think\n\nI think",
        "\n\nI think\n\n\n\nI think",
        "Answer:",
        "Answer: I think",
        " I think",
        "I think",
        "I think that",
        "I think ",
        "### Preferred Output is ",
        "### Preferred Output is 1.",
        "### Preferred Output is 2.",
    ]:
        print([text], tokenizer.encode(text))
