# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_omni.diffusion.worker.execution_slice import (
    StepwiseDiffusionExecutionSliceSessionFactory,
)


class _Runner:
    def __init__(self) -> None:
        self.released: list[str] = []

    def prepare_explicit_request(self, request):
        return SimpleNamespace(
            request=request,
            request_id=request.request_id,
            step_index=0,
            denoise_completed=False,
            extra={},
        )

    def execute_explicit_denoising_step(self, state) -> None:
        state.step_index += 1
        state.denoise_completed = state.step_index >= 2

    def prepare_explicit_vae_blocks(self, state) -> None:
        state.extra["vae_block_index"] = 0

    def supports_explicit_vae_blocks(self) -> bool:
        return True

    def execute_explicit_vae_block(self, state) -> bool:
        state.extra["vae_block_index"] += 1
        return state.extra["vae_block_index"] >= 2

    def finalize_explicit_vae_blocks(self, state):
        return f"output:{state.request_id}"

    def release_step_requests(self, request_ids) -> None:
        self.released.extend(request_ids)


def test_stepwise_factory_exposes_opaque_monotonic_slices() -> None:
    runner = _Runner()
    factory = StepwiseDiffusionExecutionSliceSessionFactory(runner)
    session = factory.create_session(
        "scheduler-request",
        SimpleNamespace(request_id="model-request"),
    )

    phases = []
    tags = []
    results = []
    while True:
        spec = session.next_slice()
        phases.append(spec.metadata["phase"])
        tags.append(spec.tags)
        results.append(session.execute_slice())
        if results[-1].request_done:
            break

    assert [result.sequence_no for result in results] == list(range(5))
    assert phases == [
        "preparing",
        "denoising",
        "denoising",
        "vae_blocks",
        "vae_blocks",
    ]
    assert tags[1] == tags[2] == frozenset(
        ("component:denoiser", "phase:sampling")
    )
    assert tags[3] == tags[4] == frozenset(
        ("component:vae", "phase:decode", "granularity:block")
    )
    assert [result.tags for result in results] == tags
    assert results[-1].output == "output:model-request"


def test_stepwise_sessions_keep_request_state_independent() -> None:
    factory = StepwiseDiffusionExecutionSliceSessionFactory(_Runner())
    first = factory.create_session("first", SimpleNamespace(request_id="a"))
    second = factory.create_session("second", SimpleNamespace(request_id="b"))

    assert first.execute_slice().sequence_no == 0
    assert first.execute_slice().metadata["phase"] == "denoising"
    assert second.execute_slice().sequence_no == 0
    assert second.next_slice().sequence_no == 1
    assert first.next_slice().sequence_no == 2


def test_stepwise_factory_falls_back_to_one_decode_slice() -> None:
    runner = _Runner()
    runner.supports_explicit_vae_blocks = lambda: False
    runner.execute_explicit_decode = (
        lambda state: f"decoded:{state.request_id}"
    )
    session = StepwiseDiffusionExecutionSliceSessionFactory(
        runner
    ).create_session("request", SimpleNamespace(request_id="model-request"))

    assert session.execute_slice().metadata["phase"] == "preparing"
    assert session.execute_slice().metadata["phase"] == "denoising"
    assert session.execute_slice().metadata["phase"] == "denoising"
    result = session.execute_slice()
    assert result.metadata["phase"] == "decode"
    assert result.tags == frozenset(("component:vae", "phase:decode"))
    assert result.request_done is True
    assert result.output == "decoded:model-request"
