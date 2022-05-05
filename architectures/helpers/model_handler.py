from architectures.helpers.constants import selected_model
from architectures.convmixer import get_cm_model
from architectures.vision_transformer import get_vit_model
from architectures.mlp_mixer import get_mm_model


def get_model():
    if selected_model == "convmixer":
        return get_cm_model()
    elif selected_model == "vision_transformer":
        return get_vit_model()
    elif selected_model == "mlp_mixer":
        return get_mm_model()
    elif selected_model == "cnn_ta":
        pass
