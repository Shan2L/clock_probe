"""Pure helpers for discovering Ray physical-node topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

HEAD_RESOURCE = "node:__internal_head__"


@dataclass(frozen=True)
class RayNode:
    """Normalized metadata for one alive physical Ray node."""

    node_id: str
    name: str
    address: str
    is_head: bool
    resources: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return data suitable for Ray transport and JSON encoding."""
        return asdict(self)


def discover_alive_nodes(ray_node_records: Sequence[dict[str, Any]]) -> list[RayNode]:
    """Normalize ``ray.nodes()`` records and identify the unique head."""
    nodes: list[RayNode] = []
    for record in ray_node_records:
        if not record.get("Alive", False):
            continue
        resources = {
            str(name): float(value)
            for name, value in record.get("Resources", {}).items()
        }
        nodes.append(
            RayNode(
                node_id=str(record["NodeID"]),
                name=str(
                    record.get("NodeName")
                    or record.get("NodeManagerHostname")
                    or record.get("NodeManagerAddress")
                ),
                address=str(record["NodeManagerAddress"]),
                is_head=resources.get(HEAD_RESOURCE, 0.0) > 0,
                resources=resources,
            )
        )

    if not nodes:
        raise RuntimeError("Ray reports no alive physical nodes")
    head_nodes = [node for node in nodes if node.is_head]
    if len(head_nodes) != 1:
        raise RuntimeError(
            "Expected exactly one Ray head resource "
            f"{HEAD_RESOURCE!r}, found {len(head_nodes)}"
        )
    return sorted(nodes, key=lambda node: (not node.is_head, node.name))


def get_head_node(nodes: Sequence[RayNode]) -> RayNode:
    """Return the unique Ray head from normalized nodes."""
    head_nodes = [node for node in nodes if node.is_head]
    if len(head_nodes) != 1:
        raise RuntimeError(f"Expected one Ray head node, found {len(head_nodes)}")
    return head_nodes[0]
