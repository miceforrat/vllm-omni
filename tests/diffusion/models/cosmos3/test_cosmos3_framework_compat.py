# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import nullcontext
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_FRAMEWORK_AVAILABLE = find_spec("cosmos_framework") is not None
requires_cosmos_framework = pytest.mark.skipif(
    not _FRAMEWORK_AVAILABLE,
    reason="cosmos_framework source is not present on PYTHONPATH",
)


def test_cosmos3_exposes_native_step_execution_protocol() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )
    from vllm_omni.diffusion.models.interface import (
        SupportsStepExecution,
        supports_step_execution,
    )

    # 协议检查不需要加载权重；这里只验证 Cosmos3 已接入 Omni 原生接口。
    pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)

    assert pipeline.supports_step_execution is True
    assert supports_step_execution(pipeline) is True
    assert isinstance(pipeline, SupportsStepExecution)


def test_cosmos3_vae_blocks_match_unsplit_temporal_decode() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    class FakeVAE:
        dtype = torch.float32
        use_tiling = False
        tile_sample_min_height = 64
        tile_sample_min_width = 64
        spatial_compression_ratio = 8
        config = SimpleNamespace(scaling_factor=1.0, patch_size=None)

        def __init__(self) -> None:
            self._feat_map = []
            self._conv_idx = [0]

        def clear_cache(self) -> None:
            self._feat_map = []

        def _execution_context(self):
            return nullcontext()

        @staticmethod
        def post_quant_conv(value):
            return value + 1

        @staticmethod
        def decoder(value, **_kwargs):
            return value * 2

        def decode(self, value, return_dict=False):
            projected = self.post_quant_conv(value)
            chunks = [
                self.decoder(projected[:, :, index : index + 1])
                for index in range(projected.shape[2])
            ]
            output = torch.cat(chunks, dim=2)
            return (output,) if not return_dict else output

    pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
    object.__setattr__(pipeline, "vae", FakeVAE())
    latents = torch.randn(1, 4, 4, 2, 2)
    state = SimpleNamespace(latents=latents.clone(), extra={})

    expected = pipeline.vae.decode(latents, return_dict=False)[0]
    pipeline.prepare_vae_blocks(state)
    observed_boundaries = []
    while True:
        observed_boundaries.append(state.extra["vae_block_index"])
        if pipeline.run_vae_block(state):
            break
    actual = torch.cat(state.extra["vae_decoded_chunks"], dim=2)

    assert observed_boundaries == [0, 1, 2, 3]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_framework_preflight_checks_every_policy_symbol(monkeypatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import utils as cosmos3_utils

    imported = []

    def record_import(module_name: str, symbol_name: str, error_message: str):
        imported.append((module_name, symbol_name, error_message))
        return object()

    monkeypatch.setattr(cosmos3_utils, "lazy_import", record_import)

    cosmos3_utils.preflight_cosmos3_action_framework_imports()

    expected_symbols = {
        "ActionTransformPipeline",
        "FlowUniPCMultistepScheduler",
        "build_abs_pose_from_components",
        "convert_rotation",
        "get_domain_id",
        "pose_abs_to_rel",
        "pose_rel_to_abs",
    }
    assert {symbol_name for _, symbol_name, _ in imported} == expected_symbols
    assert [(module_name, symbol_name) for module_name, symbol_name, _ in imported] == list(
        cosmos3_utils._ROBOLAB_FRAMEWORK_IMPORTS.values()
    )


def test_lazy_import_reports_framework_symbol_skew(monkeypatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import utils as cosmos3_utils

    monkeypatch.setattr(cosmos3_utils, "import_module", lambda _module_name: object())

    with pytest.raises(ImportError, match="MissingSymbol") as exc_info:
        cosmos3_utils.lazy_import(
            "cosmos_framework.fake_module",
            "MissingSymbol",
            "Cosmos Framework compatibility error.",
        )

    assert isinstance(exc_info.value.__cause__, AttributeError)


@requires_cosmos_framework
def test_action_domain_table_matches_current_framework_and_legacy_aliases() -> None:
    from cosmos_framework.data.generator.action.utils.domain_utils import (
        EMBODIMENT_TO_DOMAIN_ID as FRAMEWORK_DOMAIN_IDS,
    )

    from vllm_omni.diffusion.models.cosmos3.action import EMBODIMENT_TO_DOMAIN_ID as VLLM_DOMAIN_IDS
    from vllm_omni.diffusion.models.cosmos3.utils import _LEGACY_EMBODIMENT_ALIASES

    canonical_vllm_domain_ids = {
        name: domain_id for name, domain_id in VLLM_DOMAIN_IDS.items() if name not in _LEGACY_EMBODIMENT_ALIASES
    }

    assert canonical_vllm_domain_ids == FRAMEWORK_DOMAIN_IDS
    assert {alias: VLLM_DOMAIN_IDS[alias] for alias in _LEGACY_EMBODIMENT_ALIASES} == {
        alias: FRAMEWORK_DOMAIN_IDS[canonical_name] for alias, canonical_name in _LEGACY_EMBODIMENT_ALIASES.items()
    }


@requires_cosmos_framework
def test_current_cosmos_framework_policy_modules_import_and_instantiate() -> None:
    from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
    from cosmos_framework.data.generator.action.utils.pose_utils import (
        build_abs_pose_from_components,
        convert_rotation,
        pose_abs_to_rel,
        pose_rel_to_abs,
    )
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import (
        FlowUniPCMultistepScheduler,
    )

    transform = ActionTransformPipeline(
        max_action_dim=64,
        cfg_dropout_rate=0.0,
        format_prompt_as_json=True,
    )
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(4, device=torch.device("cpu"), shift=5.0)

    assert transform.prompt_json_formatter is not None
    assert len(scheduler.timesteps) == 4
    assert get_domain_id("droid_lerobot") == 8
    assert convert_rotation([0.0, 0.0, 0.0, 1.0], "quat_xyzw", "rot6d").shape == (6,)
    assert callable(pose_abs_to_rel)
    assert callable(pose_rel_to_abs)
    assert callable(build_abs_pose_from_components)


@requires_cosmos_framework
def test_real_action_transform_pipeline_matches_droid_policy_contract() -> None:
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    transform = ActionTransformPipeline(
        max_action_dim=64,
        cfg_dropout_rate=0.0,
        format_prompt_as_json=True,
    )
    sample = {
        "ai_caption": "Pick up the cube.",
        "video": torch.zeros((3, 3, 192, 320), dtype=torch.uint8),
        "action": torch.zeros((3, 8), dtype=torch.float32),
        "conditioning_fps": torch.tensor(15, dtype=torch.long),
        "mode": "wam",
        "domain_id": torch.tensor(8, dtype=torch.long),
        "viewpoint": "concat_view",
        "additional_view_description": "Wrist view above two exterior views.",
    }

    transformed = transform(sample, "256")

    assert isinstance(transformed["ai_caption"], dict)
    assert transformed["action"].shape == (3, 64)
    assert int(transformed["raw_action_dim"].item()) == 8
    assert transformed["sequence_plan"].has_action is True
    assert transformed["mode"] == "wam"
    assert int(transformed["domain_id"].item()) == 8
