MLP_BASELINE_CONFIG = {
    "hidden_sizes": (128, 64),
    "dropout": 0.0,
    "batch_size": 256,
    "learning_rate": 0.001,
}

MLP_SEARCH_SPACE = {
    "hidden_sizes": [
        (64, 32),
        (128, 64),
        (256, 128),
        (256, 128, 64),
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
}