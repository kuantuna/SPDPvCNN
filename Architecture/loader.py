import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

image_size = 11
auto = tf.data.AUTOTUNE
batch_size = 16

imageList = np.load("C:\\Users\\Tuna\\Desktop\\2022-spring\\CS402\\SPDPvCNN\\ETF\\Images.npy")
labelList = np.load("C:\\Users\\Tuna\\Desktop\\2022-spring\\CS402\\SPDPvCNN\\ETF\\Labels.npy")

x_train, x_test, y_train, y_test = train_test_split(imageList, labelList, test_size=0.1, random_state=41)
val_split = 0.1

val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
x_val, y_val = x_train[:val_indices], y_train[:val_indices]

data_augmentation = keras.Sequential(
    [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"), ],
    name="data_augmentation",
)

def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)


train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)

reconstructed_model = keras.models.load_model("C:\\Users\\Tuna\\Desktop\\Saved Models\\model-best-33.h5", custom_objects={'MyOptimizer': tfa.optimizers.AdamW})
predictions = reconstructed_model.predict(test_dataset)
classes = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_test, classes)
print(cm)
cr = classification_report(y_test, classes)
print(cr)
f1 = f1_score(y_test, classes, average='micro')
print(f1)