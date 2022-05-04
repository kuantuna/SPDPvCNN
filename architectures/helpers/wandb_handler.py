import wandb
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import selected_model
from architectures.helpers.constants import threshold

hyperparameters = hyperparameters[selected_model]


def initialize_wandb():
    if selected_model == "convmixer":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "filters": hyperparameters["filters"],
                       "depth": hyperparameters["depth"],
                       "kernel_size": hyperparameters["kernel_size"],
                       "patch_size": hyperparameters["patch_size"],
                       "threshold": f"0.{threshold}",
                       "image_size": hyperparameters["image_size"]
                   })
    elif selected_model == "vision_transformer":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "projection_dim": hyperparameters["projection_dim"],
                       "num_heads": hyperparameters["num_heads"],
                       "patch_size": hyperparameters["patch_size"],
                       "transformer_layers": hyperparameters["transformer_layers"],
                       "threshold": f"0.{threshold}",
                   })
    elif selected_model == "mlp_mixer":
        pass
    elif selected_model == "cnn_ta":
        pass
