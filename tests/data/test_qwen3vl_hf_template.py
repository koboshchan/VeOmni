"""CPU contract tests against the actual public Qwen3-VL tokenizer."""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from veomni.data.chat_template import Qwen3VLChatTemplate
from veomni.utils.constants import IGNORE_INDEX, TYPE2INDEX


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


@pytest.mark.parametrize(
    "action", ["<|action_start|><use_absolute>0 0 0 ; W A ; ; ; ; ; ;<|action_end|>", "  W A  ", ""]
)
def test_native_ids_and_assistant_loss(tokenizer, action):
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|action_start|>", "<|action_end|>", "<use_absolute>"]}
    )
    template = Qwen3VLChatTemplate(SimpleNamespace(tokenizer=tokenizer))
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    messages = [
        {"role": "user", "content": image, "loss_mask": 0},
        {"role": "assistant", "content": action, "loss_mask": 1},
    ]
    hf_messages = [
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": action},
    ]
    expected = tokenizer.apply_chat_template(
        hf_messages, tokenize=True, return_dict=False, add_generation_prompt=False
    )
    actual = template.tokenize_messages(messages)
    assert actual["input_ids"].tolist() == expected
    prompt = tokenizer.apply_chat_template(
        hf_messages[:1], tokenize=True, return_dict=False, add_generation_prompt=True
    )
    assert actual["labels"].tolist() == [IGNORE_INDEX] * len(prompt) + expected[len(prompt) :]
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert end_id in actual["labels"][len(prompt) :]
    text = tokenizer.decode(expected)
    assert text == "<|im_start|>user\n" + image + "<|im_end|>\n<|im_start|>assistant\n" + action + "<|im_end|>\n"
    assert "<think>" not in text and "<|im_start|>system" not in text

    encoded = template.encode_messages([["user", ("image", None)], ["assistant", ("text", action)]], {"image": [1]})
    restored = encoded["input_ids"].clone()
    restored[restored == TYPE2INDEX["input"]["image"]] = template.image_token_id
    assert restored.tolist() == expected
    assert torch.equal(encoded["labels"], actual["labels"])


def test_multiturn_masks_and_explicit_system(tokenizer):
    template = Qwen3VLChatTemplate(SimpleNamespace(tokenizer=tokenizer))
    messages = [
        {"role": "system", "content": "Explicit only.", "loss_mask": 0},
        {"role": "user", "content": "  first\n", "loss_mask": 0},
        {"role": "assistant", "content": "", "loss_mask": 0},
        {"role": "user", "content": "second", "loss_mask": 0},
        {"role": "assistant", "content": "  done  ", "loss_mask": 1},
    ]
    hf = [{k: m[k] for k in ("role", "content")} for m in messages]
    actual = template.tokenize_messages(messages)
    expected = tokenizer.apply_chat_template(hf, tokenize=True, return_dict=False, add_generation_prompt=False)
    start = len(tokenizer.apply_chat_template(hf[:-1], tokenize=True, return_dict=False, add_generation_prompt=True))
    assert actual["input_ids"].tolist() == expected
    assert actual["labels"].tolist() == [IGNORE_INDEX] * start + expected[start:]
