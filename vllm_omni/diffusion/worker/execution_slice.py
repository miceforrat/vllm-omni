# SPDX-License-Identifier: Apache-2.0
"""Diffusion Runner 的模型无关 execution-slice session factory。"""

from __future__ import annotations

from typing import Any

from vllm.experimental.execution_slice import (
    ExecutionSliceDescriptor,
    ExecutionSliceResult,
    ExecutionSliceSession,
    ExecutionSliceSessionFactory,
)


class StepwiseDiffusionExecutionSliceSession:
    """把 Omni step-mode Runner 适配为请求私有的 slice session。

    本类只依赖 Runner 的 step-mode 接口，不依赖具体 pipeline 类。模型阶段
    仅作为 metadata 暴露，外部调度器看到的始终是单调递增的 execution slice。
    """

    def __init__(self, runner: Any, request_id: str, request: Any) -> None:
        self._runner = runner
        self._request_id = str(request_id)
        self._request = request
        self._state: Any | None = None
        self._phase = "preparing"
        self._sequence_no = 0
        self._vae_blocks_prepared = False
        self._closed = False

    @property
    def request_id(self) -> str:
        return self._request_id

    def _phase_index(self) -> int | None:
        state = self._state
        if self._phase == "denoising" and state is not None:
            return int(state.step_index)
        if self._phase == "vae_blocks" and state is not None:
            return int(state.extra.get("vae_block_index", 0))
        return None

    def _metadata(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "phase_index": self._phase_index(),
        }

    def _tags(self) -> frozenset[str]:
        """返回稳定的执行性质；同类迭代 step 共享同一组 tag。"""
        if self._phase == "preparing":
            return frozenset(("component:preparation", "phase:prepare"))
        if self._phase == "denoising":
            return frozenset(("component:denoiser", "phase:sampling"))
        if self._phase == "vae_blocks":
            return frozenset(
                ("component:vae", "phase:decode", "granularity:block")
            )
        if self._phase == "decode":
            return frozenset(("component:vae", "phase:decode"))
        raise RuntimeError(f"Unknown diffusion execution phase {self._phase!r}")

    def next_slice(self) -> ExecutionSliceDescriptor:
        if self._closed:
            raise RuntimeError(
                f"Execution slice session {self.request_id!r} is closed"
            )
        return ExecutionSliceDescriptor(
            sequence_no=self._sequence_no,
            metadata=self._metadata(),
            tags=self._tags(),
        )

    def execute_slice(self) -> ExecutionSliceResult:
        spec = self.next_slice()
        metadata = dict(spec.metadata)

        if self._phase == "preparing":
            self._state = self._runner.prepare_explicit_request(self._request)
            self._request = None
            self._phase = "denoising"
            self._sequence_no += 1
            return ExecutionSliceResult(
                spec.sequence_no, False, metadata=metadata, tags=spec.tags
            )

        state = self._state
        if state is None:
            raise RuntimeError(
                f"Execution slice session {self.request_id!r} has no state"
            )

        if self._phase == "denoising":
            self._runner.execute_explicit_denoising_step(state)
            if state.denoise_completed:
                supports_blocks = getattr(
                    self._runner, "supports_explicit_vae_blocks", None
                )
                self._phase = (
                    "vae_blocks"
                    if callable(supports_blocks) and supports_blocks()
                    else "decode"
                )
            self._sequence_no += 1
            return ExecutionSliceResult(
                spec.sequence_no, False, metadata=metadata, tags=spec.tags
            )

        if self._phase == "decode":
            output = self._runner.execute_explicit_decode(state)
            self._state = None
            self._closed = True
            self._sequence_no += 1
            return ExecutionSliceResult(
                spec.sequence_no,
                True,
                output,
                metadata,
                spec.tags,
            )

        if self._phase == "vae_blocks":
            if not self._vae_blocks_prepared:
                self._runner.prepare_explicit_vae_blocks(state)
                self._vae_blocks_prepared = True
            done = self._runner.execute_explicit_vae_block(state)
            output = None
            if done:
                output = self._runner.finalize_explicit_vae_blocks(state)
                self._state = None
                self._closed = True
            self._sequence_no += 1
            return ExecutionSliceResult(
                spec.sequence_no,
                bool(done),
                output,
                metadata,
                spec.tags,
            )

        raise RuntimeError(f"Unknown diffusion execution phase {self._phase!r}")

    def close(self) -> None:
        if self._closed:
            return
        state = self._state
        self._state = None
        self._closed = True
        if state is not None:
            self._runner.release_step_requests([state.request_id])


class StepwiseDiffusionExecutionSliceSessionFactory:
    """为任意支持 Omni step-mode 接口的 Runner 创建请求级 session。"""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def create_session(
        self,
        request_id: str,
        work: Any,
    ) -> ExecutionSliceSession:
        return StepwiseDiffusionExecutionSliceSession(
            self._runner,
            request_id,
            work,
        )


def create_execution_slice_session_factory(
    runner: Any,
) -> ExecutionSliceSessionFactory:
    """选择 Runner 自定义 Session Factory，默认使用 step-mode 实现。"""
    factory = getattr(runner, "create_execution_slice_session_factory", None)
    if callable(factory):
        session_factory = factory()
        if session_factory is None:
            raise RuntimeError(
                "Runner returned no execution-slice session factory"
            )
        return session_factory
    # ExecUnit 在模型加载前创建 factory；step-mode 能力由 Runner 在真正创建
    # session、调用 prepare_explicit_request 时校验。
    return StepwiseDiffusionExecutionSliceSessionFactory(runner)


__all__ = [
    "StepwiseDiffusionExecutionSliceSessionFactory",
    "StepwiseDiffusionExecutionSliceSession",
    "create_execution_slice_session_factory",
]
