import numpy as np

spec = {
    "vertices": {
        "input": {
            "num_nodes": 1,
            "activation": lambda x: x,
            "bias": 0
        },
        "memory_in0": {
            "num_nodes": 1,
            "activation": lambda x: x,
            "bias": 0
        },
        "memory_out0": {
            "num_nodes": 1,
            "activation": lambda x: x,
            "bias": 0
        },
        "output": {
            "num_nodes": 1,
            "activation": lambda x: x,
            "bias": 0
        }
    },
    "edges": [
        {
            "source_id": "memory_in0",
            "target_id": "output",
            "weights": np.array([[0.5]])
        },
        {
            "source_id": "input",
            "target_id": "output",
            "weights": np.array([[0.5]])
        },
        {
            "source_id": "memory_in0",
            "target_id": "memory_out0",
            "weights": np.array([[0]])
        },
        {
            "source_id": "input",
            "target_id": "memory_out0",
            "weights": np.array([[1]])
        }
    ],
    "input": "input",
    "output": "output",
    "memories": {
        "memory_in0": "memory_out0"
    }
}
