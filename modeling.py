import os
import cv2
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount('/content/drive')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
np.set_printoptions(suppress=True)

GRAY_CROPPED_POSITIVES_FOLDER_PATH = '/content/drive/MyDrive/Bundesliga/PositiveFrames/frames_gray_dg/'

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


def main():
    model = create_model(with_dropout=True)
    model = compile_model(model, lr=0.000001) # 6 decimals
    model = train_model(model, epochs=10000, patience=50)
    return model


if __name__ == '__main__':
    main()
