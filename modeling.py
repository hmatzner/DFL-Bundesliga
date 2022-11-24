import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

GRAY_CROPPED_POSITIVES_FOLDER_PATH = '/Bundesliga/PositiveFrames/frames_gray_dg/'

img_gen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    )

class_mode = 'sparse'
seed = 42
target_size = (224, 224)
batch_size = 64

train_data = img_gen.flow_from_directory(
    GRAY_CROPPED_POSITIVES_FOLDER_PATH + 'train',
    class_mode=class_mode,
    batch_size=batch_size,
    target_size=target_size,
    seed=seed
    )

valid_data = img_gen.flow_from_directory(
    GRAY_CROPPED_POSITIVES_FOLDER_PATH + 'val',
    class_mode=class_mode,
    batch_size=batch_size,
    target_size=target_size,
    seed=seed
    )

test_data = img_gen.flow_from_directory(
    GRAY_CROPPED_POSITIVES_FOLDER_PATH + 'test',
    class_mode=class_mode,
    batch_size=batch_size,
    target_size=target_size,
    seed=seed
    )

filecount = list()
FOLDER = GRAY_CROPPED_POSITIVES_FOLDER_PATH

for set_ in os.listdir(FOLDER):
    for class_ in os.listdir(os.path.join(FOLDER, set_)):
        filecount.append((set_, class_, len(os.listdir(os.path.join(FOLDER, set_, class_)))))


def create_model(with_dropout=False):
  """
  Creates the model's architecture
  @with_dropout: either to add a Dropout layer or not (boolean)
  """
  vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3))
  if with_dropout:
    model = Sequential([
        vgg_model,
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])
  else:
    model = Sequential([
        vgg_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])

  return model


def compile_model(model, lr=0.001):
  """
  Compiles the model
  @model: keras model to train
  @lr: learning rate (float)
  """
  optimizer = Adam(learning_rate=lr)

  model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics='accuracy',
              )
  return model


def train_model(model, train_data=train_data, valid_data=valid_data, epochs=100, batch_size=16, patience=5,
                account_weights=True):
    """
    Trains the model
    @model: keras model to train
    @train_data: training data (DirectoryIterator object)
    @valid_data: validation data (DirectoryIterator object)
    @epochs: number of epochs to perform for training (int)
    @batch_size: number of samples that will be propagated through the network (int)
    @patience: number of epochs to wait when using early stopping as callback (int)
    @account_weights: either to assign a weight to each class to address imbalance dataset or not (boolean)
    """
    callback = EarlyStopping(monitor='val_loss',
                             patience=patience,
                             restore_best_weights=True)

    if account_weights:
        train_events = dict()
        for file in filecount:
            if file[0] == 'train':
                train_events[file[1]] = file[2]

        total_train_samples = sum(train_events.values())

        class_weights = dict()
        ordered_events = ['challenge', 'play', 'throwin']
        for i, event in enumerate(ordered_events):
            class_weights[i] = total_train_samples / train_events[event]


    else:
        class_weights = None

    model.fit(train_data,
              validation_data=valid_data,
              batch_size=batch_size,
              epochs=epochs,
              class_weight=class_weights,
              steps_per_epoch=len(train_data),
              validation_steps=len(valid_data),
              callbacks=[callback])

    return model


def main():
    model = create_model(with_dropout=True)
    model = compile_model(model, lr=0.000001)
    model = train_model(model, epochs=10000, patience=50)
    return model


if __name__ == '__main__':
    main()
