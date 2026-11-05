# SPDX-License-Identifier: Apache-2.0
"""Optional resource providers consumed by the diffusion model runner."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol


class DiffusionMemoryProvider(Protocol):
    def allocation_context(self, tag: str) -> AbstractContextManager[Any]: ...


class DiffusionStreamProvider(Protocol):
    def execution_context(self) -> AbstractContextManager[Any]: ...


class DiffusionCommunicatorProvider(Protocol):
    def communication_context(self) -> AbstractContextManager[Any]: ...


class DiffusionPhaseProvider(Protocol):
    def install(self, pipeline: Any) -> None: ...

    def request_context(
        self, requests: Any | None = None
    ) -> AbstractContextManager[Any]: ...

    def uninstall(self) -> None: ...


@dataclass(frozen=True)
class DiffusionRuntimeProviders:
    """Model-scoped resource providers supplied by the owning runtime."""

    memory: DiffusionMemoryProvider | None = None
    stream: DiffusionStreamProvider | None = None
    communicator: DiffusionCommunicatorProvider | None = None
    phase: DiffusionPhaseProvider | None = None
