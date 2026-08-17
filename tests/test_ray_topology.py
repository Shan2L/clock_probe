import unittest

from clock_probe.ray_topology import discover_alive_nodes, get_head_node


class RayTopologyTest(unittest.TestCase):
    def test_discovers_head_and_workers(self) -> None:
        nodes = discover_alive_nodes(
            [
                {
                    "NodeID": "worker-id",
                    "NodeName": "cse-ai-9",
                    "NodeManagerAddress": "10.67.93.244",
                    "Alive": True,
                    "Resources": {"CPU": 64},
                },
                {
                    "NodeID": "head-id",
                    "NodeName": "cse-ai-6",
                    "NodeManagerAddress": "10.67.91.123",
                    "Alive": True,
                    "Resources": {
                        "CPU": 64,
                        "node:__internal_head__": 1,
                    },
                },
                {
                    "NodeID": "dead-id",
                    "NodeManagerAddress": "10.0.0.3",
                    "Alive": False,
                    "Resources": {},
                },
            ]
        )

        self.assertEqual(len(nodes), 2)
        self.assertEqual(get_head_node(nodes).node_id, "head-id")
        self.assertTrue(nodes[0].is_head)

    def test_requires_exactly_one_head(self) -> None:
        with self.assertRaises(RuntimeError):
            discover_alive_nodes(
                [
                    {
                        "NodeID": "worker-id",
                        "NodeManagerAddress": "10.0.0.2",
                        "Alive": True,
                        "Resources": {"CPU": 1},
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
