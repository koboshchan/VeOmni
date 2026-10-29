"""Opt-in Lumine vocabulary preparation, before wrapping or optimizer creation."""

import torch


LUMINE_CUSTOM_TOKENS = (
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


def prepare_lumine_vocabulary(model, tokenizer) -> int:
    """Add ordinary action tokens and grow (never shrink) the model vocabulary.

    Meta-init leaves the new rows empty until the checkpoint loader fills them.
    Eager builds already hold pretrained rows, so fill appended rows here.
    In either case the appended rows use the source table's mean, identically
    on all ranks. Existing tokenizer IDs and padded model rows are preserved.
    """
    if tokenizer is None:
        raise ValueError("model.add_custom_tokens requires a tokenizer.")
    tokenizer.add_tokens([token for token in LUMINE_CUSTOM_TOKENS if token not in tokenizer.get_vocab()])
    embeddings = model.get_input_embeddings()
    old_size = embeddings.weight.shape[0]
    target_size = max(old_size, len(tokenizer))
    if target_size > old_size:
        # HF's covariance-based mean resizing cannot run on a meta tensor.
        model.resize_token_embeddings(target_size, mean_resizing=False)
        with torch.no_grad():
            for layer in (model.get_input_embeddings(), model.get_output_embeddings()):
                if layer is not None and not layer.weight.is_meta:
                    layer.weight[old_size:].copy_(layer.weight[:old_size].float().mean(dim=0).to(layer.weight.dtype))

    # Exact parameter names, not substring matching. The shared loader otherwise
    # keeps its strict shape checks; this is only an append-only vocabulary opt-in.
    weights = {
        id(layer.weight)
        for layer in (model.get_input_embeddings(), model.get_output_embeddings())
        if layer is not None
    }
    model._vocab_expansion_parameters = {
        name for name, parameter in model.named_parameters(remove_duplicate=False) if id(parameter) in weights
    }
    return target_size
