import tensorflow as tf
import random
import types
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class DataProvider:
    def get(self):
        num_training = 45000
        num_validation = 5000
        onehot_encoder = OneHotEncoder(sparse=False)
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0 
        y_test = onehot_encoder.fit_transform(y_test)
        y_train = onehot_encoder.fit_transform(y_train)
        x_train = x_train.reshape(*x_train.shape[:1], -1)
        x_test = x_test.reshape(*x_test.shape[:1], -1)
        
        validationMask = range(num_training, num_training + num_validation)
        data = types.SimpleNamespace()
        data.train = types.SimpleNamespace()
        data.train.images = x_train[range(num_training)]
        data.train.labels = y_train[range(num_training)]
        data.train.num_examples = data.train.images.shape[0]
        
        def next_batch(number):
            indices = np.random.choice(range(data.train.num_examples), number, replace=False)
            return [data.train.images[indices], data.train.labels[indices]]
        
        data.train.next_batch = next_batch
        
        data.validation = types.SimpleNamespace()
        data.validation.images = x_train[validationMask]
        data.validation.labels = y_train[validationMask]
        data.validation.num_examples = data.validation.images.shape[0]
        data.test = types.SimpleNamespace()
        data.test.images = x_test
        data.test.labels = y_test
        data.test.num_examples = data.test.images.shape[0]

        return data

    def getSingleImageFromTrainingSet(self):
        data = self.get()
        index = random.randint(0, data.train.num_examples)
        return (data.train.labels[index], data.train.images[index].reshape(32,32,3))