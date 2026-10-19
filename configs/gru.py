GRU_BASELINE_CONFIG = {
    "hidden_size": 128,
    "number_of_layers": 1,
    "dense_size": 64,
    "dropout": 0.0,
    "batch_size": 256,
    "learning_rate": 0.001,
    "gradient_clip_norm": 1.0,
}


GRU_SEARCH_SPACE = {
    "hidden_size": [
        64,
        128,
        256,
    ],
    "number_of_layers": [
        1,
        2,
    ],
    "dense_size": [
        32,
        64,
        128,
    ],
    "dropout": [
        0.0,
        0.1,
        0.2,
        0.3,
    ],
    "batch_size": [
        256,
        512,
        1024,
        2048,
    ],
    "learning_rate": [
        0.0001,
        0.0005,
        0.001,
    ],
    "gradient_clip_norm": [
        1.0,
    ],
}