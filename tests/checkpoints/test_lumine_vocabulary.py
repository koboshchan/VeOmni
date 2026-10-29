"""Lumine vocabulary expansion preserves pretrained weights on CPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from tests.tools.training_utils import unbuilt_runtime
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.models.checkpoint.weights import (
    _dispatch_parameter,
    init_empty_weights,
    load_model_weights,
    rank0_load_and_broadcast_weights,
)
from veomni.models.custom_tokens import LUMINE_CUSTOM_TOKENS, prepare_lumine_vocabulary
from veomni.trainer.vlm_trainer import VLMMModelArguments, VLMModelRuntime


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "base": 1, "old": 2, "tokens": 3}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


def model(vocab_size=4, tied=False):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            tie_word_embeddings=tied,
        )
    )


@pytest.fixture
def cpu_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        yield _init_parallel_state(dp_size=1, dp_mode="ddp")
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.parametrize("tied", [False, True])
def test_eager_resize_preserves_ids_rows_and_tied_weights(tied):
    network, tok = model(tied=tied), tokenizer()
    original_ids = tok.get_vocab()
    before = network.get_input_embeddings().weight.detach().clone()
    size = prepare_lumine_vocabulary(network, tok)

    assert size == len(tok) == 4 + len(LUMINE_CUSTOM_TOKENS)
    assert {key: tok.convert_tokens_to_ids(key) for key in original_ids} == original_ids
    for token in LUMINE_CUSTOM_TOKENS:
        assert tok.encode(token, add_special_tokens=False) == [tok.convert_tokens_to_ids(token)]
    torch.testing.assert_close(network.get_input_embeddings().weight[:4], before, rtol=0, atol=0)
    torch.testing.assert_close(
        network.get_input_embeddings().weight[4:],
        before.mean(0).expand(size - 4, -1),
    )
    if tied:
        assert network.get_input_embeddings().weight is network.get_output_embeddings().weight
    first_parameter = network.get_input_embeddings().weight
    prepare_lumine_vocabulary(network, tok)
    assert network.get_input_embeddings().weight is first_parameter


def test_existing_padded_vocab_is_never_shrunk():
    network, tok = model(vocab_size=64), tokenizer()
    before = network.get_input_embeddings().weight
    assert prepare_lumine_vocabulary(network, tok) == 64
    assert network.get_input_embeddings().weight is before
    assert network.config.vocab_size == 64


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_meta_resize_load_and_export_roundtrip(tmp_path, cpu_group, broadcast, tied):
    source = model(tied=tied)
    source.save_pretrained(tmp_path / "base")
    with init_empty_weights():
        expanded = model(tied=tied)
    tok = tokenizer()
    prepare_lumine_vocabulary(expanded, tok)
    assert expanded.get_input_embeddings().weight.is_meta

    loader = rank0_load_and_broadcast_weights if broadcast else load_model_weights
    loader(expanded, str(tmp_path / "base"), init_device="cpu")
    expected = source.state_dict()
    for name, value in expanded.state_dict().items():
        if name in expanded._vocab_expansion_parameters:
            torch.testing.assert_close(value[:4], expected[name], rtol=0, atol=0)
            torch.testing.assert_close(value[4:], expected[name].mean(0).expand(value.shape[0] - 4, -1))
        else:
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert torch.isfinite(value).all()

    expanded.save_pretrained(tmp_path / "export")
    tok.save_pretrained(tmp_path / "export")
    reloaded = Qwen3ForCausalLM.from_pretrained(tmp_path / "export")
    reloaded_tok = PreTrainedTokenizerFast.from_pretrained(tmp_path / "export")
    assert reloaded.config.vocab_size == len(reloaded_tok)
    assert reloaded_tok.get_vocab() == tok.get_vocab()
    for name, value in expanded.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[name], rtol=0, atol=0)


def test_full_table_is_expanded_before_dtensor_factory(cpu_group):
    mesh = cpu_group.dp_mesh
    network = nn.Linear(2, 6, bias=False)
    network._vocab_expansion_parameters = {"weight"}
    network.weight = nn.Parameter(distribute_tensor(torch.empty(6, 2), mesh, [Shard(0)]))
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    seen = []

    def factory(full, device_mesh, placements):
        seen.append(full.shape)
        return distribute_tensor(full, device_mesh, placements)

    with torch.no_grad():
        _dispatch_parameter(network, "weight", source, dtensor_factory=factory)
    assert seen == [torch.Size((6, 2))]
    torch.testing.assert_close(network.weight.full_tensor()[:4], source)
    torch.testing.assert_close(network.weight.full_tensor()[4:], source.mean(0).expand(2, -1))


@pytest.mark.parametrize("shape", [(7, 2), (4, 3), (0, 2)])
def test_opt_in_does_not_silently_truncate_or_accept_wrong_hidden_size(shape):
    network = nn.Linear(2, 6, bias=False)
    network._vocab_expansion_parameters = {"weight"}
    with pytest.raises(ValueError, match="Cannot expand vocabulary"):
        _dispatch_parameter(network, "weight", torch.empty(shape))


def test_non_opted_in_parameter_keeps_strict_loading():
    network = nn.Linear(2, 6, bias=False)
    with pytest.raises(RuntimeError):
        _dispatch_parameter(network, "weight", torch.ones(4, 2))


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_prepares_assets_once_before_wrap(monkeypatch, enabled):
    network, tok = model(), tokenizer()
    calls = []
    monkeypatch.setattr("veomni.trainer.vlm_trainer.build_foundation_model", lambda **kwargs: network)
    monkeypatch.setattr("veomni.models.auto.build_processor", lambda *args, **kwargs: calls.append("processor") or tok)
    args = VLMMModelArguments(model_path="unused", add_custom_tokens=enabled)
    runtime = unbuilt_runtime(args, cls=VLMModelRuntime)
    runtime._validate_torch_compile = Mock()
    runtime._build_model()
    assert len(calls) == int(enabled)
    assert network.get_input_embeddings().num_embeddings == (len(tok) if enabled else 4)
    runtime._build_model_assets()
    assert calls == ["processor"]
    assert runtime.tokenizer is tok
    assert runtime.model_assets == [network.config, tok]


def test_adding_tokens_requires_a_tokenizer():
    with pytest.raises(ValueError, match="requires a tokenizer"):
        prepare_lumine_vocabulary(model(), None)


@pytest.mark.parametrize("option", ["lora_config", "ep_sharded_stream_load"])
def test_runtime_rejects_unsupported_partial_loading(monkeypatch, option):
    monkeypatch.setattr("veomni.trainer.vlm_trainer.build_foundation_model", lambda **kwargs: model())
    args = VLMMModelArguments(model_path="unused", add_custom_tokens=True)
    setattr(args, option, SimpleNamespace() if option == "lora_config" else True)
    runtime = unbuilt_runtime(args, cls=VLMModelRuntime)
    with pytest.raises(ValueError, match="requires full tuning"):
        runtime._build_model()


def _two_rank_load_worker(rank, path, broadcast, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2)
    try:
        parallel_state = _init_parallel_state(dp_size=2, dp_mode="fsdp2")
        with init_empty_weights():
            expanded = model()
        prepare_lumine_vocabulary(expanded, tokenizer())
        # CPU DTensor reproduces pre-sharded FSDP2 parameter placement without
        # requiring a GPU forward/backward or an accelerator allocation.
        for name in expanded._vocab_expansion_parameters:
            parent_name, _, leaf = name.rpartition(".")
            parent = expanded.get_submodule(parent_name)
            parameter = parent._parameters[leaf]
            parent._parameters[leaf] = nn.Parameter(
                distribute_tensor(torch.empty(parameter.shape), parallel_state.dp_mesh, [Shard(0)])
            )
        loader = rank0_load_and_broadcast_weights if broadcast else load_model_weights
        loader(expanded, path, init_device="cpu", dtensor_factory=distribute_tensor)
        original = Qwen3ForCausalLM.from_pretrained(path)
        for name in expanded._vocab_expansion_parameters:
            actual = expanded.get_parameter(name).full_tensor()
            source = original.get_parameter(name)
            torch.testing.assert_close(actual[:4], source, rtol=0, atol=0)
            torch.testing.assert_close(actual[4:], source.mean(0).expand(actual.shape[0] - 4, -1))
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("broadcast", [False, True])
def test_two_rank_load_expands_full_vocabulary_before_sharding(tmp_path, broadcast):
    model().save_pretrained(tmp_path)
    spawn_with_rendezvous(_two_rank_load_worker, args=(str(tmp_path), broadcast), nprocs=2)
