# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest

import vllm.envs as envs
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import CachedHfTokenizer
from vllm.tokenizers.registry import (
    TokenizerRegistry,
    get_tokenizer,
    resolve_tokenizer_args,
)


class TestTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TestTokenizer":
        return TestTokenizer(path_or_repo_id)  # type: ignore

    def __init__(self, path_or_repo_id: str | Path) -> None:
        super().__init__()

        self.path_or_repo_id = path_or_repo_id

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def pad_token_id(self) -> int:
        return 2

    @property
    def is_fast(self) -> bool:
        return True


@pytest.mark.parametrize("runner_type", ["generate", "pooling"])
def test_resolve_tokenizer_args_idempotent(runner_type):
    tokenizer_mode, tokenizer_name, args, kwargs = resolve_tokenizer_args(
        "facebook/opt-125m",
        runner_type=runner_type,
    )

    assert (tokenizer_mode, tokenizer_name, args, kwargs) == resolve_tokenizer_args(
        tokenizer_name, *args, **kwargs
    )


def test_customized_tokenizer():
    TokenizerRegistry.register("test_tokenizer", __name__, TestTokenizer.__name__)

    tokenizer = TokenizerRegistry.load_tokenizer("test_tokenizer", "abc")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.path_or_repo_id == "abc"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2

    tokenizer = get_tokenizer("abc", tokenizer_mode="test_tokenizer")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.path_or_repo_id == "abc"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2


def test_fastokens_patch_applies_to_hf_tokenizer(monkeypatch):
    import vllm.tokenizers.fastokens as fastokens_module
    import vllm.tokenizers.hf as hf_module

    sentinel = object()
    calls: dict[str, object] = {}

    def fake_apply_fastokens_patch() -> None:
        calls["patched"] = True

    def fake_from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ):
        calls["args"] = args
        calls["kwargs"] = kwargs
        calls["path_or_repo_id"] = path_or_repo_id
        calls["trust_remote_code"] = trust_remote_code
        calls["revision"] = revision
        calls["download_dir"] = download_dir
        return sentinel

    monkeypatch.setattr(envs, "VLLM_USE_FASTOKENS", True, raising=False)
    monkeypatch.setattr(
        fastokens_module,
        "apply_fastokens_patch",
        fake_apply_fastokens_patch,
    )
    monkeypatch.setattr(
        hf_module.AutoTokenizer,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    monkeypatch.setattr(hf_module, "get_cached_tokenizer", lambda tokenizer: sentinel)
    monkeypatch.setattr(
        hf_module,
        "get_sentence_transformer_tokenizer_config",
        lambda *_args, **_kwargs: {},
    )

    tokenizer = CachedHfTokenizer.from_pretrained(
        "abc",
        trust_remote_code=True,
        revision="main",
        download_dir="/tmp/cache",
    )

    assert tokenizer is sentinel
    assert calls == {
        "patched": True,
        "args": (),
        "kwargs": {"use_fast": True},
        "path_or_repo_id": "abc",
        "trust_remote_code": True,
        "revision": "main",
        "download_dir": "/tmp/cache",
    }


def test_apply_fastokens_patch_is_idempotent(monkeypatch):
    import importlib
    import sys
    import types

    fastokens_module = importlib.import_module("vllm.tokenizers.fastokens")

    calls: dict[str, int] = {"patched": 0}

    def fake_patch_transformers() -> None:
        calls["patched"] += 1

    fake_fastokens = types.ModuleType("fastokens")
    fake_fastokens.patch_transformers = fake_patch_transformers

    monkeypatch.setitem(sys.modules, "fastokens", fake_fastokens)
    monkeypatch.setattr(fastokens_module, "_patched", False, raising=False)
    monkeypatch.setattr(
        fastokens_module,
        "version",
        lambda _name: "0.2.0",
    )

    fastokens_module.apply_fastokens_patch()
    fastokens_module.apply_fastokens_patch()

    assert calls["patched"] == 1
