
# Import dependencies
import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Splitting the data



# Compiling model
# resnet_model = tf.keras.Sequential()
# resnet = tf.keras.applications.resnet50.ResNet50()
# resnet_model.add(resnet)

# resnet_model.add(tf.keras.layers.Flatten())

# # 128 neurons because our input data is not a lot
# resnet_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# resnet_model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

# resnet_model.compile(optimizer=tf.keras.optimizers.SGD(
#     learning_rate=1e-1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model



# Evaluating Model
