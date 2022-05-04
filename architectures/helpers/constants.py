hyperparameters = {
    "convmixer": {
        "learning_rate": "WarmUpCosine",  # 0.01
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 500,
        "filters": 256,
        "depth": 8,
        "kernel_size": 7,
        "patch_size": 5,
        "image_size": 67,
    },
    "vision_transformer": {
        "learning_rate": "WarmUpCosine",  # 0.001
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 150,
        "image_size": 67,  # We'll resize input images to this size
        "patch_size": 7,  # Size of the patches to be extract from the input images
        "projection_dim": 64,  # 128
        "num_heads": 4,
        "transformer_layers": 8,  # 6, 8, 10
        "mlp_head_units": [2048, 1024],
        "num_classes": 3,
        "input_shape": (67, 67, 1)
    },
    "mlp_mixer": {

    },
    "cnn_ta": {

    }
}

hyperparameters["vision_transformer"]["num_patches"] = (
    hyperparameters["vision_transformer"]["image_size"] // hyperparameters["vision_transformer"]["patch_size"]) ** 2
hyperparameters["vision_transformer"]["transformer_units"] = [
    hyperparameters["vision_transformer"]["projection_dim"] * 2,
    hyperparameters["vision_transformer"]["projection_dim"],
]

# selected_model = "convmixer"
selected_model = "vision_transformer"
# selected_model = "mlp_mixer"
# selected_model = "cnn_ta"

etf_list = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
threshold = "01"
