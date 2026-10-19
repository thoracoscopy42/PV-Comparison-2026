TRANSFORMER_BASELINE_CONFIG = {
    "d_model": 128,
    "number_of_heads": 8,
    "number_of_layers": 2,
    "dim_feedforward": 512,
    "dropout": 0.0,
    "batch_size": 256,
    "learning_rate": 0.0005,
    "gradient_clip_norm": 1.0,
    "weight_decay": 0.0001,
}


TRANSFORMER_SEARCH_SPACE = {
    "d_model": [
        64,
        128,
        256,
    ],
    "number_of_heads": [
        4,
        8,
    ],
    "number_of_layers": [
        1,
        2,
        3,
    ],
    "dim_feedforward": [
        128,
        256,
        512,
    ],
    "dropout": [
        0.0,
        0.1,
        0.2,
    ],
    "batch_size": [
        256,
        512,
        1024,
    ],
    "learning_rate": [
        0.0001,
        0.0005,
        0.001,
    ],
    "gradient_clip_norm": [
        1.0,
    ],
    "weight_decay": [
    0.0,
    0.00001,
    0.0001,
    0.001,
    ],
}