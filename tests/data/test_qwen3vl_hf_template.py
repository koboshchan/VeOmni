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


@pytest.mark.parametrize("action", ["<|action_start|><|use_abs|>0 0 0 ; W A ; ; ; ; ; ;<|action_end|>", "  W A  ", ""])
def test_native_ids_and_assistant_loss(tokenizer, action):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|action_start|>", "<|action_end|>", "<|use_abs|>"]})
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


@pytest.mark.parametrize("recording_frame", [False, True])
def test_image_data_transform_masks_and_native_parity(tokenizer, tmp_path, recording_frame, monkeypatch):
    from PIL import Image
    from transformers import AutoProcessor

    from veomni.data.data_transform import _process_sample_qwen_vl_base
    from veomni.models.custom_tokens import LUMINE_CUSTOM_TOKENS

    # The pipeline's opt-in tokens are ordinary added tokens, not stripped specials.
    tokenizer.add_tokens([token for token in LUMINE_CUSTOM_TOKENS if token not in tokenizer.get_vocab()])
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    processor.tokenizer = tokenizer
    if recording_frame:
        real_processor = processor.image_processor

        class CheckedProcessor:
            patch_size = real_processor.patch_size
            merge_size = real_processor.merge_size

            def __call__(self, images, **kwargs):
                assert kwargs["do_resize"] is False
                assert images[0].size == (1280, 736)
                assert images[0].crop((0, 0, 1280, 720)).getextrema() == ((255, 255), (0, 0), (0, 0))
                assert images[0].crop((0, 720, 1280, 736)).getextrema() == ((0, 0), (0, 0), (0, 0))
                return real_processor(images=images, **kwargs)

        monkeypatch.setattr(processor, "image_processor", CheckedProcessor())
    template = Qwen3VLChatTemplate(processor)
    action = "<|action_start|><|use_abs|>0 0 0 ; W ;<|action_end|>"
    for token in ("<|action_start|>", "<|action_end|>", "<|use_abs|>"):
        assert len(tokenizer.encode(token, add_special_tokens=False)) == 1

    def position_ids(**kwargs):
        return {"position_ids": torch.arange(kwargs["input_ids"].numel()).view(1, 1, -1)}

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 720) if recording_frame else (56, 56), color=(255, 0, 0)).save(image_path)
    sample = {
        "source": "lumine_pretrain",
        "images": [str(image_path)],
        "text": action,
    }
    kwargs = (
        dict(image_frame_width=1280, image_frame_height=720, image_min_pixels=921600, image_max_pixels=921600)
        if recording_frame
        else {}
    )
    output = _process_sample_qwen_vl_base(sample, processor, template, position_ids, **kwargs)[0]
    if recording_frame:
        assert output["image_grid_thw"].tolist() == [[1, 46, 80]]
        assert int(output["image_mask"].sum()) == 920
        # Native 1280x720 content plus 16 black rows, not a stretched 736px image.
        patches = output["pixel_values"].reshape(1, 23, 40, 2, 2, 3, 2, 16, 16)
        assert (patches[..., 0, :, :, :] == 1).any()
        assert (patches == -1).any()
    count = int(output["image_mask"].sum())
    assert count > 0
    hf = [
        {"role": "user", "content": "<|vision_start|>" + "<|image_pad|>" * count + "<|vision_end|>"},
        {"role": "assistant", "content": action},
    ]
    expected = tokenizer.apply_chat_template(hf, tokenize=True, return_dict=False, add_generation_prompt=False)
    restored = output["input_ids"].clone()
    assert (restored[output["image_mask"]] == 0).all()
    restored[output["image_mask"]] = template.image_token_id
    assert restored.tolist() == expected
    prefix = tokenizer.apply_chat_template(hf[:1], tokenize=True, return_dict=False, add_generation_prompt=True)
    assert output["labels"].tolist() == [IGNORE_INDEX] * len(prefix) + expected[len(prefix) :]
