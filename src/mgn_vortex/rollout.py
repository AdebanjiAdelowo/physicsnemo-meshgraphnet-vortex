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

# Adapted from examples/cfd/vortex_shedding_mgn/inference.py of NVIDIA PhysicsNeMo, commit
# b45a5c810c741e6b41f8515be24c51121f8fc21f (MGNRollout.predict). Modified by Adebanji
# Adelowo (2026): the loop is a function over one trajectory of an already constructed
# dataset and model, it does not modify the batch in place, it can also run with the
# reference state as input at every step (one-step predictions), and it returns arrays
# instead of storing lists for the animation. The normalisation, the boundary mask and
# the time integration are unchanged.

"""Autoregressive rollout and one-step prediction with a trained MeshGraphNet."""

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader


@torch.no_grad()
def rollout(model, dataset, trajectory: int, device, autoregressive: bool = True, stride: int = 1) -> dict:
    """Predict one test trajectory.

    Parameters
    ----------
    model : callable
        ``model(node_features, edge_features, graph)`` returning normalised targets.
    dataset : VortexSheddingDataset
        A ``valid`` or ``test`` split (items are ``(graph, cells, rollout_mask)``).
    trajectory : int
        Index of the trajectory in the dataset.
    autoregressive : bool
        True: the predicted velocity is the next input (free-running rollout from the
        initial condition). False: every input is the reference state (one-step predictions).
    stride : int
        Evaluate every ``stride``-th step. Only allowed for one-step predictions.

    Returns
    -------
    dict of NumPy arrays: ``pred`` and ``exact`` of shape (steps, nodes, 3) holding
    (u, v, p) at the time levels ``steps`` (the level reached by each prediction, the
    initial condition being level 0), ``input_velocity`` (steps, nodes, 2), ``mesh_pos``,
    ``cells`` and the boolean ``mask`` of nodes whose velocity is advanced by the model.
    """
    if autoregressive and stride != 1:
        raise ValueError("a rollout cannot skip steps")
    per_trajectory = dataset.num_steps - 1
    start = trajectory * per_trajectory
    indices = range(start, start + per_trajectory, stride)
    dataloader = PyGDataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, drop_last=False)

    stats = {key: value.to(device) for key, value in dataset.node_stats.items()}
    pred, exact, inputs = [], [], []
    for i, (graph, cells, mask) in enumerate(dataloader):
        graph = graph.to(device)
        # denormalize data
        x_velocity = dataset.denormalize(graph.x[:, 0:2], stats["velocity_mean"], stats["velocity_std"])
        y_velocity = dataset.denormalize(graph.y[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"])
        y_pressure = dataset.denormalize(graph.y[:, [2]], stats["pressure_mean"], stats["pressure_std"])

        # inference step
        velocity = pred[-1][:, 0:2] if (autoregressive and i > 0) else x_velocity
        invar = graph.x.clone()
        invar[:, 0:2] = dataset.normalize_node(velocity, stats["velocity_mean"], stats["velocity_std"])
        pred_i = model(invar, graph.edge_attr, graph).detach()  # predict

        # denormalize prediction
        pred_velocity = dataset.denormalize(pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"])
        pred_pressure = dataset.denormalize(pred_i[:, [2]], stats["pressure_mean"], stats["pressure_std"])

        # the velocity is advanced only on interior and outflow nodes (node types 0 and 5);
        # inflow and wall nodes keep their input value
        mask = mask.reshape(-1, 1).to(device)
        pred_velocity = torch.where(mask, pred_velocity, torch.zeros_like(pred_velocity))

        # integration
        pred.append(torch.cat((pred_velocity + velocity, pred_pressure), dim=-1))
        exact.append(torch.cat((y_velocity + x_velocity, y_pressure), dim=-1))
        inputs.append(x_velocity)

    return {
        "pred": torch.stack(pred).cpu().numpy(),
        "exact": torch.stack(exact).cpu().numpy(),
        "input_velocity": torch.stack(inputs).cpu().numpy(),
        "steps": np.asarray(indices) - start + 1,
        "mesh_pos": graph["mesh_pos"].cpu().numpy(),
        "cells": torch.squeeze(cells, 0).numpy(),
        "mask": mask.reshape(-1).cpu().numpy(),
    }
