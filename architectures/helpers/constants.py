from architectures.helpers.warmup_cosine import WarmUpCosine
from architectures.helpers.one_cycle import OneCycleLRScheduler
from tensorflow import keras
import matplotlib.pyplot as plt

selected_model = "convmixer"
# selected_model = "convmixer_tf"
# selected_model = "vision_transformer"
# selected_model = "mlp_mixer"
# selected_model = "cnn_ta"

etf_list = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
threshold = "01"

hyperparameters = {
    "convmixer": {
        "learning_rate_type": "Not found",  # "WarmUpCosine"
        "weight_decay": 0.0001,
        "batch_size": 32,
        "num_epochs": 200,
        "filters": 256,
        "depth": 8,
        "kernel_size": 9,
        "patch_size": 3,
        "image_size": 67,
    },
    "convmixer_tf": {
        "learning_rate_type": 0.1,  # "WarmUpCosine"
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 500,
        "filters": 256,
        "depth": 8,
        "kernel_size": 7,
        "patch_size": 7,
        "image_size": 67,
    },
    "vision_transformer": {
        "learning_rate_type": "Not found",  # 0.001
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 150,
        "image_size": 67,  # We'll resize input images to this size
        "patch_size": 5,  # Size of the patches to be extract from the input images
        "projection_dim": 64,  # 128
        "num_heads": 4,
        "transformer_layers": 8,  # 6, 8, 10
        "mlp_head_units": [2048, 1024],
        "num_classes": 3,
    },
    "mlp_mixer": {
        "learning_rate_type": "Not found",  # "WarmUpCosine", ReduceLROnPlateau
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 500,
        "dropout_rate": 0.5,
        "image_size": 67,  # We'll resize input images to this size.
        "patch_size": 7,  # Size of the patches to be extract from the input images
        "embedding_dim": 256,  # Number of hidden units.
        "num_blocks": 4,  # Number of blocks.
        "num_classes": 3,
    },
    "cnn_ta": {
        "learning_rate_type": "Not found",
        "batch_size": 128,
        "num_epochs": 500,
        "first_dropout_rate": 0.25,
        "second_dropout_rate": 0.5,
        "kernel_size": 5,
        "image_size": 67,  # We'll resize input images to this size.
        "num_classes": 3,
    }
}


if hyperparameters[selected_model]["learning_rate_type"] == "WarmUpCosine":
    TOTAL_STEPS = int(
        (50000 / hyperparameters[selected_model]["batch_size"]) * hyperparameters[selected_model]["num_epochs"])
    WARMUP_STEPS = 10000
    INIT_LR = 0.001
    WAMRUP_LR = 0.0001

    scheduled_lrs = WarmUpCosine(
        learning_rate_base=INIT_LR,
        total_steps=TOTAL_STEPS,
        warmup_learning_rate=WAMRUP_LR,
        warmup_steps=WARMUP_STEPS,
    )
    hyperparameters[selected_model]["learning_rate"] = scheduled_lrs
#     lrs = [scheduled_lrs(step) for step in range(TOTAL_STEPS)]
#     plt.plot(lrs)
#     plt.xlabel("Step", fontsize=14)
#     plt.ylabel("LR", fontsize=14)
#     plt.grid()
#     plt.show()
else:
    hyperparameters[selected_model]["learning_rate"] = hyperparameters[selected_model]["learning_rate_type"]
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.6, patience=10, verbose=1
    # )
    # hyperparameters[selected_model]["learning_rate_scheduler"] = reduce_lr
    # one_cycle_lr_scheduler = OneCycleLRScheduler(hyperparameters[selected_model]["num_epochs"], hyperparameters[selected_model]["learning_rate"],
    #                                              32465 / hyperparameters[selected_model]["batch_size"])
    # hyperparameters[selected_model]["learning_rate_scheduler"] = one_cycle_lr_scheduler


hyperparameters["convmixer"]["input_shape"] = (
    hyperparameters["convmixer"]["image_size"], hyperparameters["convmixer"]["image_size"], 1)

hyperparameters["convmixer_tf"]["input_shape"] = (
    hyperparameters["convmixer_tf"]["image_size"], hyperparameters["convmixer_tf"]["image_size"], 1)

hyperparameters["vision_transformer"]["num_patches"] = (
    hyperparameters["vision_transformer"]["image_size"] // hyperparameters["vision_transformer"]["patch_size"]) ** 2
hyperparameters["vision_transformer"]["transformer_units"] = [
    hyperparameters["vision_transformer"]["projection_dim"] * 2,
    hyperparameters["vision_transformer"]["projection_dim"],
]
hyperparameters["vision_transformer"]["input_shape"] = (
    hyperparameters["vision_transformer"]["image_size"], hyperparameters["vision_transformer"]["image_size"], 1)

hyperparameters["mlp_mixer"]["num_patches"] = (
    hyperparameters["mlp_mixer"]["image_size"] // hyperparameters["mlp_mixer"]["patch_size"]) ** 2

hyperparameters["mlp_mixer"]["input_shape"] = (
    hyperparameters["mlp_mixer"]["image_size"], hyperparameters["mlp_mixer"]["image_size"], 1)

hyperparameters["cnn_ta"]["input_shape"] = (
    hyperparameters["cnn_ta"]["image_size"], hyperparameters["cnn_ta"]["image_size"], 1)
