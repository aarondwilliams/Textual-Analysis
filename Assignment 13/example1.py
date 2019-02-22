import numpy as np

np.random.seed(31)
spec = {
    "vertices": {
        "input": {
            "num_nodes": 2,
            "activation": lambda x: x,
            "bias": 0
        },
        "memory_in0": {
            "num_nodes": 2,
            "activation": lambda x: x,
            "bias": 0
        },
        "h1": {
            "num_nodes": 5,
            "activation": lambda x: 1 / (1 + np.exp(-x)),
            "bias": 0
        },
        "h2": {
            "num_nodes": 9,
            "activation": lambda x: 1 / (1 + np.exp(-x)),
            "bias": 0
        },
        "memory_out0": {
            "num_nodes": 2,
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
            "target_id": "h1",
            "weights": np.random.normal(size=(2, 5))
        },
        {
            "source_id": "input",
            "target_id": "h1",
            "weights": np.random.normal(size=(2, 5))
        },
        {
            "source_id": "h1",
            "target_id": "h2",
            "weights": np.random.normal(size=(5, 9))
        },
        {
            "source_id": "h1",
            "target_id": "memory_out0",
            "weights": np.random.normal(size=(5, 2))
        },
        {
            "source_id": "h2",
            "target_id": "output",
            "weights": np.random.normal(size=(9, 1))
        }
    ],
    "input": "input",
    "output": "output",
    "memories": {
        "memory_in0": "memory_out0"
    }
}
