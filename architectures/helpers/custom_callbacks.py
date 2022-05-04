import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import selected_model
from architectures.helpers.constants import threshold


hyperparameters = hyperparameters["convmixer"]
# Runs at the end of every epoch and prints the confusion matrix


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, epoch_counter, time, y_test) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.epoch_counter = epoch_counter
        self.time = time
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.test_dataset)
        classes = np.argmax(predictions, axis=1)
        print(confusion_matrix(self.y_test, classes))

        export_path_keras = f"SavedModels/{selected_model}/{threshold}/{int(self.time)}-{hyperparameters['filters']}x{hyperparameters['depth']}-k{hyperparameters['kernel_size']}p{hyperparameters['patch_size']}e{self.epoch_counter}.h5"
        self.model.save_weights(export_path_keras)
        self.epoch_counter += 1
