# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from examples/cfd/vortex_shedding_mgn/train.py of NVIDIA PhysicsNeMo, commit
# b45a5c810c741e6b41f8515be24c51121f8fc21f (the MeshGraphNet constructor call of
# MGNTrainer.__init__). Modified by Adebanji Adelowo (2026): processor_size is passed
# explicitly instead of being left at its default.

"""Construction of the PhysicsNeMo MeshGraphNet. The network itself is imported, not copied."""

import torch
from omegaconf import DictConfig
from physicsnemo.models.meshgraphnet import MeshGraphNet


def build_model(cfg: DictConfig, processor_size: int) -> MeshGraphNet:
    """MeshGraphNet as configured upstream, with ``processor_size`` message-passing blocks."""
    return MeshGraphNet(
        cfg.num_input_features,
        cfg.num_edge_features,
        cfg.num_output_features,
        processor_size=processor_size,
        # MGN with recompute_activation currently supports only SiLU activation function.
        mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
        do_concat_trick=cfg.do_concat_trick,
        num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
        recompute_activation=cfg.recompute_activation,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
