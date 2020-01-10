import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import classification_report
import pandas as pd
from sklearn.datasets import load_iris
import keras

def visualize_training(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()

    # A chart showing our training vs validation loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()


def B2_CNN(tr_X, tr_Y, te_X, te_Y, val_X, val_Y):

    training_images = tr_X / 255.0
    training_labels = tr_Y
    test_images = te_X / 255.0
    test_labels = te_Y
    val_images = val_X / 255.0
    val_labels = val_Y

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.0001, decay=0.0000006, momentum=0.9, nesterov=False)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(training_images, training_labels, batch_size=128, validation_data=(test_images, test_labels),
                        epochs=100)

    res = model.evaluate(x=test_images, y=test_labels, batch_size=128, verbose=1, sample_weight=None, steps=None)
    predictions = model.predict(test_images, batch_size=128, verbose=0, steps=None)
    print('The classification accuracy on the test set is:', res[1])
    print(predictions)
    y_pred = np.argmax(predictions, axis=1)
    test_labelsn = np.argmax(test_labels, axis=1)
    print(np.argmax(test_labels, axis=1))
    print(y_pred)
    print(classification_report(test_labelsn, y_pred))

    visualize_training(history)