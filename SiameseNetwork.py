import tensorflow as tf
import numpy as np
import sklearn.model_selection
from typing import Callable


BATCH_SIZE = 32
ORIG_IMAGE_SIZE = (250, 250, 3)
INPUT_IMAGE_SIZE = (105, 105, 3)
TRAIN_LIST_PATH = 'train.txt'
TEST_LIST_PATH = 'train.txt'
NUM_EPOCHS = 10


def create_features_vector() -> tf.keras.Model:
    conv_layers_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)
    conv_bias_init = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)
    dense_layer_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.2)
    dense_bias_init = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)

    conv_regularize = tf.keras.regularizers.L2()
    dense_regularize = tf.keras.regularizers.L2()

    cnn = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (10, 10), padding='valid', strides=1, activation='relu',
                               kernel_initializer=conv_layers_init, bias_initializer=conv_bias_init,
                               kernel_regularizer=conv_regularize),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(128, (7, 7), padding='valid', strides=1, activation='relu',
                               kernel_initializer=conv_layers_init, bias_initializer=conv_bias_init,
                               kernel_regularizer=conv_regularize),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(128, (4, 4), padding='valid', strides=1, activation='relu',
                               kernel_initializer=conv_layers_init, bias_initializer=conv_bias_init,
                               kernel_regularizer=conv_regularize),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(256, (4, 4), padding='valid', strides=1, activation='relu',
                               kernel_initializer=conv_layers_init, bias_initializer=conv_bias_init,
                               kernel_regularizer=conv_regularize),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='sigmoid', kernel_initializer=dense_layer_init,
                              bias_initializer=dense_bias_init, kernel_regularizer=dense_regularize)
    ], name='cnn_feature')
    return cnn


def siamese_model(cnn_model: tf.keras.Model, distance_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor])\
        -> tf.keras.models.Model:
    im1 = tf.keras.layers.Input(INPUT_IMAGE_SIZE)
    im2 = tf.keras.layers.Input(INPUT_IMAGE_SIZE)

    features1 = cnn_model(im1)
    features2 = cnn_model(im2)

    distance_layer = tf.keras.layers.Lambda(lambda features: distance_function(features[0], features[1]))
    distance = distance_layer([features1, features2])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    model = tf.keras.models.Model([im1, im2], output)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=200, decay_rate=0.99)  # TODO try other optimizers
    # optimizer = tf.optimizers.SGD(momentum=0.5, learning_rate=lr_schedule)
    optimizer = tf.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def l1_distance(features1: tf.Tensor, features2: tf.Tensor) -> tf.Tensor:
    return tf.abs(features1 - features2)


def l2_distance(features1: tf.Tensor, features2: tf.Tensor) -> tf.Tensor:
    return tf.abs(features1 - features2)


def get_im_path(name: str, num: str) -> str:
    return 'lfw2\\' + name + '\\' + name + '_' + '0' * (4 - len(num)) + num + '.jpg'


def read_image(image_path: str) -> tf.Tensor:
    image_string = tf.io.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.image.resize(image, INPUT_IMAGE_SIZE[:2])
    return image


def build_set(im_list_path: str) -> tuple[np.ndarray, np.ndarray]:
    im1_paths = []
    im2_paths = []
    with open(im_list_path, 'r') as train_file:
        num_matches = int(train_file.readline().strip())
        lines = train_file.readlines()
        labels = np.concatenate([np.ones(num_matches), np.zeros(len(lines) - num_matches)])
        for line in lines[:num_matches]:
            name, num1, num2 = line.strip().split()
            im1_paths.append(get_im_path(name, num1))
            im2_paths.append(get_im_path(name, num2))
        for line in lines[num_matches:]:
            name1, num1, name2, num2 = line.strip().split()
            im1_paths.append(get_im_path(name1, num1))
            im2_paths.append(get_im_path(name2, num2))
    x = np.array([(read_image(im1_paths[i]), read_image(im2_paths[i])) for i in range(len(labels))])
    y = np.array(labels).astype('float32').reshape(-1, 1)
    return x, y
    # dataset = tf.data.Dataset.from_tensor_slices(((im1_paths, im2_paths), labels))
    # # dataset = tf.data.Dataset.zip(
    # #     (tf.data.Dataset.from_tensor_slices(im1_paths),
    # #      tf.data.Dataset.from_tensor_slices(im2_paths),
    # #      tf.data.Dataset.from_tensor_slices(labels)))
    # dataset.shuffle(len(labels))
    # # dataset.map(lambda im1_path, im2_path, label: ((read_image(im1_path), read_image(im2_path)), label))
    # dataset.map(lambda im_paths, label: ((read_image(im_paths[0]), read_image(im_paths[1])), label))
    # # dataset.map(parse_data)
    # dataset.batch(BATCH_SIZE)
    # dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # return dataset

x, y = build_set(TRAIN_LIST_PATH)
x, x_validation, y, y_validation = sklearn.model_selection.train_test_split(x, y, train_size=0.8, shuffle=True)
x_test, y_test = build_set(TEST_LIST_PATH)

cnn_model = create_features_vector()

model = siamese_model(cnn_model, l2_distance)
model.fit(x=(x[:, 0, :, :], x[:, 1, :, :]), y=y,
          validation_data=((x_validation[:, 0, :, :], x_validation[:, 1, :, :]), y_validation),
          batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
model.evaluate(x=(x_test[:, 0, :, :], x_test[:, 1, :, :]), y=y_test, )
