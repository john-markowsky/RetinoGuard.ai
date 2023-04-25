import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.optimizers import Adam

# Create a model based on the DenseNet121 architecture
def create_model():
    input_shape = (224, 224, 3)
    base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)

    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))

    optimizer = Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# # Create the model
# model = create_model()

# # Save the model as an HDF5 file
# model.save("model.h5")