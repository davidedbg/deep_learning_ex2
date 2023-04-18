import tensorflow as tf
import numpy as np

IMAGE_SIZE = (250, 250)
TRAIN_LIST_PATH = 'train.txt'
TEST_LIST_PATH = 'train.txt'

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
        tf.keras.layers.Dense(4096, activation='sigmoid', kernel_initializer=dense_layer_init,
                              bias_initializer=dense_bias_init, kernel_regularizer=dense_regularize)
    ], name='cnn_feature')
    return cnn


def siamese_model(cnn_model: tf.keras.Model, distance_function) -> tf.keras.models.Model:
    im1 = tf.keras.layers.Input(IMAGE_SIZE)
    im2 = tf.keras.layers.Input(IMAGE_SIZE)

    features1 = cnn_model(im1)
    features2 = cnn_model(im2)

    distance_layer = tf.keras.layers.Lambda(lambda features: distance_function(features[0], features[1]))
    distance = distance_layer([features1, features2])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    model = tf.keras.models.Model(inputs=[im1, im2], output=output)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=200, decay_rate=0.99)  # TODO try other optimizers
    optimizer = tf.optimizers.SGD(momentum=0.5, learning_rate=lr_schedule)
    model.compile(loss='binary_cross_entropy', optimizer=optimizer)
    return model


def l1_distance(features1: tf.Tensor, features2: tf.Tensor) -> tf.Tensor:
    return tf.abs(features1 - features2)


def l2_distance(features1: tf.Tensor, features2: tf.Tensor):
    return tf.abs(features1 - features2)


def get_im_path(name: str, num: str) -> str:
    return name + '/' + name + '_' + '0' * (4 - len(num)) + num + '.jpg'


def read_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path) / 255
    return image


def build_set(im_list_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(im1_paths),
                                   tf.data.Dataset.from_tensor_slices(im2_paths),
                                   tf.data.Dataset.from_tensor_slices(labels)))
    dataset.shuffle(len(labels))
    dataset.map(lambda im1_path, im2_path, label: ((read_image(im1_path), read_image(im2_path)), label))
    return dataset


train_set = build_set(TRAIN_LIST_PATH)
test_set = build_set(TEST_LIST_PATH)

