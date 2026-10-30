"""Seven-only registry keeps the trained Qwen action vocabulary and chat IDs."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

# Load the vocabulary helper alone: this CPU tokenizer test needs no model registry.
_spec = importlib.util.spec_from_file_location(
    "lumine_custom_tokens", Path(__file__).parents[2] / "veomni/models/custom_tokens.py"
)
_registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_registry)

# Original training registration order, including words already in the base vocab.
_TRAINED_REGISTRY = (
    "<|action_start|>",
    "<|action_end|>",
    "<use_absolute>",
    "<|thought_start|>",
    "<|thought_end|>",
    "LB",
    "RB",
    "MB",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Esc",
    "Tab",
    "Caps",
    "Shift",
    "Ctrl",
    "Alt",
    "Space",
)
_NEW_TOKENS = (
    "<|action_start|>",
    "<|action_end|>",
    "<use_absolute>",
    "<|thought_start|>",
    "<|thought_end|>",
    "Eleven",
    "Twelve",
)


@pytest.fixture(scope="module")
def tokenizers():
    source = os.environ.get("LUMINE_BASE_TOKENIZER", "Qwen/Qwen3-VL-2B-Instruct")
    original = AutoTokenizer.from_pretrained(source)
    current = AutoTokenizer.from_pretrained(source)
    base_vocab = original.get_vocab()
    assert tuple(token for token in _TRAINED_REGISTRY if token not in base_vocab) == _NEW_TOKENS
    assert _registry.LUMINE_CUSTOM_TOKENS == _NEW_TOKENS
    assert original.add_tokens([token for token in _TRAINED_REGISTRY if token not in base_vocab]) == 7
    # Exercise production's missing-only path without allocating a 2B model.
    embeddings = SimpleNamespace(weight=SimpleNamespace(shape=(len(original),)))
    model = SimpleNamespace(
        get_input_embeddings=lambda: embeddings,
        get_output_embeddings=lambda: None,
        named_parameters=lambda **kwargs: [],
    )
    assert _registry.prepare_lumine_vocabulary(model, current) == len(original)
    current_vocab = current.get_vocab()
    assert current_vocab == original.get_vocab()
    assert {token: current_vocab[token] for token in base_vocab} == base_vocab
    return original, current


@pytest.mark.parametrize(
    "action",
    [
        "<|action_start|><use_absolute>0 1279 719 ; W A ; LB RB MB ; zero one two three four five six seven eight nine ; One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve ; Esc Tab Caps Shift Ctrl Alt Space ;<|action_end|>",
        "<|thought_start|>keep moving<|thought_end|><|action_start|>1 -23 45 ; W ; LB ; ; ; ; ;<|action_end|>",
        "<|action_start|>0 0 0 ; ; ; ; ; ; ;<|action_end|>",
    ],
)
def test_trained_action_and_full_chat_ids_unchanged(tokenizers, action):
    original, current = tokenizers
    assert current.encode(action, add_special_tokens=False) == original.encode(action, add_special_tokens=False)
    messages = [
        {"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>"},
        {"role": "assistant", "content": action},
    ]
    assert current.apply_chat_template(messages, tokenize=True, return_dict=False) == original.apply_chat_template(
        messages, tokenize=True, return_dict=False
    )
    assert current.decode(current.encode(action, add_special_tokens=False)) == action
    # Existing words retain their trained leading-space BPE, not AddedToken matching.
    for word in set(_TRAINED_REGISTRY) - set(_NEW_TOKENS):
        assert current.encode(" " + word, add_special_tokens=False) == original.encode(
            " " + word, add_special_tokens=False
        )
        assert word not in current.added_tokens_encoder


def test_saved_training_tokenizer_matches_registry(tokenizers):
    source = os.environ.get("LUMINE_SAVED_TOKENIZER")
    if source is None:
        pytest.skip("Set LUMINE_SAVED_TOKENIZER to compare an existing trained export")
    _, current = tokenizers
    saved = AutoTokenizer.from_pretrained(source)
    assert saved.get_vocab() == current.get_vocab()
    assert saved.chat_template == current.chat_template
    action = "<|action_start|><use_absolute>0 1279 719 ; W A ; LB ; Eleven Twelve ; ; ; ;<|action_end|>"
    messages = [{"role": "user", "content": "frame"}, {"role": "assistant", "content": action}]
    assert saved.apply_chat_template(messages, tokenize=True, return_dict=False) == current.apply_chat_template(
        messages, tokenize=True, return_dict=False
    )
