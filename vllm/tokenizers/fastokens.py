# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fastokens transformer patch helper."""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

_MIN_FASTOKENS_VERSION = Version("0.2.0")
_patched = False


def apply_fastokens_patch() -> None:
    global _patched
    if _patched:
        return

    try:
        fastokens_version = Version(version("fastokens"))
    except PackageNotFoundError as error:
        raise ImportError(
            "VLLM_USE_FASTOKENS=1 requires the fastokens package to be installed."
        ) from error

    if fastokens_version < _MIN_FASTOKENS_VERSION:
        raise ImportError(
            f"fastokens>={_MIN_FASTOKENS_VERSION} is required, "
            f"found {fastokens_version}."
        )

    import fastokens

    fastokens.patch_transformers()
    _patched = True
