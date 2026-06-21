from __future__ import annotations

from typing import Any


def model_kwargs_for_experiment(experiment_name: str, model_dim: int) -> dict[str, Any]:
    name = experiment_name.upper()
    kwargs: dict[str, Any] = {"model_dim": model_dim}

    if name in {"T1_B0_BASELINE", "T2_B0_GLOBAL"}:
        kwargs.update(
            use_physics_branch=False,
            use_causal_graph=False,
            use_causal_node_features=False,
            use_delay=False,
            outcome_readout_type="global",
            use_topology_features=False,
        )
    elif name == "T1_B0_PHYS_GATED":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=False,
            use_causal_node_features=False,
            use_delay=False,
            fusion_type="gated",
        )
    elif name == "T1_B0_TFCCM_NODE":
        kwargs.update(
            use_physics_branch=False,
            use_causal_graph=False,
            use_causal_node_features=True,
            use_delay=False,
        )
    elif name == "T1_B0_TFCCM_GRAPH_NO_DELAY":
        kwargs.update(
            use_physics_branch=False,
            use_causal_graph=True,
            use_causal_node_features=False,
            use_delay=False,
        )
    elif name == "T1_B0_TFCCM_GRAPH_DELAY":
        kwargs.update(
            use_physics_branch=False,
            use_causal_graph=True,
            use_causal_node_features=False,
            use_delay=True,
        )
    elif name == "T1_FULL_PGC":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            fusion_type="gated",
        )
    elif name == "T2_FULL_GLOBAL":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            outcome_readout_type="global",
            use_topology_features=False,
        )
    elif name == "T2_FULL_ATTENTION":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            outcome_readout_type="attention",
            use_topology_features=False,
        )
    elif name == "T2_FULL_ATTENTION_TOPOLOGY":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            outcome_readout_type="attention",
            use_topology_features=True,
        )
    elif name == "C1_FULL_PGC_CONCAT":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            fusion_type="concat",
        )
    elif name == "C2_FULL_PGC_RANDOM_WEIGHT":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            random_graph=True,
            random_graph_mode="weight",
        )
    elif name == "C2_FULL_PGC_RANDOM_PERMUTE":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            random_graph=True,
            random_graph_mode="permute",
        )
    elif name == "C3_FULL_PGC_UNDIRECTED":
        kwargs.update(
            use_physics_branch=True,
            use_causal_graph=True,
            use_causal_node_features=True,
            use_delay=True,
            directed_graph=False,
        )
    else:
        raise ValueError(f"Unknown experiment_name={experiment_name!r}")
    return kwargs
