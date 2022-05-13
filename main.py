#load the data
#unpack the data
#create the data generator
#create trainig set / validation set
#train the model

import src.data.load_data as loader #add them into src folder
import src.models.cnn as classifier #add them into src folder
import os
import tensorflow as tf
from tensorflow import keras

src ='https://dsti-aws-class-website-ravand.s3.eu-west-1.amazonaws.com/kagglecatsanddogs_5340.zip'
dst = '/content/dog_cat_classification/data/raw'

loader.get_data(src,dst)
loader.unzip_data(dst)

#clean the data
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(dst,"PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dog_cat_classification/data/raw/PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337, #importnat to specify the seed to make sure training and validation are exclusive (no corruption)
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dog_cat_classification/data/raw/PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


model = classifier.make_model(input_shape=image_size + (3,), num_classes=2)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
