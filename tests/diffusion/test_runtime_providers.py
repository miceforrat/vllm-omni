# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.runtime_providers import DiffusionRuntimeProviders


class _ContextProvider:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.uninstalled = False
        self.requests = None

    @contextmanager
    def _context(self):
        self.events.append(f"enter:{self.name}")
        try:
            yield
        finally:
            self.events.append(f"exit:{self.name}")

    def execution_context(self):
        return self._context()

    def communication_context(self):
        return self._context()

    def request_context(self, requests=None):
        self.requests = requests
        return self._context()

    def uninstall(self) -> None:
        self.uninstalled = True


def test_runtime_provider_contexts_are_nested_in_resource_order() -> None:
    events: list[str] = []
    communicator = _ContextProvider("communicator", events)
    stream = _ContextProvider("stream", events)
    phase = _ContextProvider("phase", events)
    runner = object.__new__(DiffusionModelRunner)
    runner.runtime_providers = DiffusionRuntimeProviders(
        communicator=communicator,
        stream=stream,
        phase=phase,
    )

    requests = [object()]
    with runner._runtime_execution_context(requests):
        events.append("execute")

    assert events == [
        "enter:communicator",
        "enter:stream",
        "enter:phase",
        "execute",
        "exit:phase",
        "exit:stream",
        "exit:communicator",
    ]
    assert phase.requests is requests


def test_release_runtime_providers_uninstalls_phase_provider() -> None:
    phase = _ContextProvider("phase", [])
    runner = object.__new__(DiffusionModelRunner)
    runner.runtime_providers = DiffusionRuntimeProviders(phase=phase)

    runner.release_runtime_providers()

    assert phase.uninstalled is True
    assert runner.runtime_providers is None
