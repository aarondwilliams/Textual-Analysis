# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 08:56:31 2018

@author: Aaron
"""

import numpy as np

spec = {
    "vertices": {
        "input": {
            "num_nodes": 13,
            "activation": lambda x: x,
            "bias": 0
        },
        "memory_in0": {
            "num_nodes": 3,
            "activation": lambda x: x,
            "bias": 0      
        },
        "multiplier": {
            "num_nodes": 2,
            "activation": lambda x: 0 if x[0] < 0 else (x[0]+1)*x[1],
            "bias": 0        
        },
        "mem_hold": {
            "num_nodes": 2,
            "activation": lambda x: [0,0] if x[0] < 0 else x,
            "bias": 0      
        },
        "memory_out0": {
            "num_nodes": 3,
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
            "source_id": "input",
            "target_id": "multiplier",
            "weights": np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                        [0,0],[0,1],[0,1],[0,1],[0,-1],[0,-1],[0,-1]])
        },
        {
            "source_id": "input",
            "target_id": "mem_hold",
            "weights": np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,1],
                        [-100,0],[-100,0],[-100,0],[-100,0],[-100,0],[-100,0]])
        },
        {
            "source_id": "memory_in0",
            "target_id": "multiplier",
            "weights": np.array([[0,0],
                        [-100,0],
                        [1,0]])
        },
        {
            "source_id": "memory_in0",
            "target_id": "memory_out0",
            "weights": np.array([[1,0,0],
                        [0,0,0],
                        [0,0,0]])
        },
        {
            "source_id": "multiplier",
            "target_id": "memory_out0",
            "weights": np.array([1,0,0])
        },
        {
            "source_id": "memory_in0",
            "target_id": "mem_hold",
            "weights": np.array([[0,0],
                        [1,0],
                        [0,1]])  
        },
        {
            "source_id": "mem_hold",
            "target_id": "memory_out0",
            "weights": np.array([[0,1,0],
                        [0,0,1]])
        },
        {
            "source_id": "memory_out0",
            "target_id": "output",
            "weights": np.array([[1],
                        [0],
                        [0]])
        }
    ],
    "input": "input",
    "output": "output",
    "memories": {
        "memory_in0": "memory_out0"
    }
}
