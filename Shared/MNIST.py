from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import types

class DataProvider:
    # Train = 55000
    # Validation = 5000
    # Test = 10000
    def get(self):
        data = input_data.read_data_sets("Data/MNIST", one_hot=True)
        
        dataTransformed = types.SimpleNamespace()
        dataTransformed.train = types.SimpleNamespace()
        dataTransformed.train.images = data.train.images
        dataTransformed.train.labels = data.train.labels
        dataTransformed.train.num_examples = data.train.num_examples
        
        def next_batch(number):
            indices = np.random.choice(range(dataTransformed.train.num_examples), number, replace=False)
            return [dataTransformed.train.images[indices], dataTransformed.train.labels[indices]]
        
        dataTransformed.train.next_batch = next_batch
        
        dataTransformed.validation = types.SimpleNamespace()
        dataTransformed.validation.images = data.validation.images
        dataTransformed.validation.labels = data.validation.labels
        dataTransformed.validation.num_examples = data.validation.num_examples
        
        dataTransformed.test = types.SimpleNamespace()
        dataTransformed.test.images = data.test.images
        dataTransformed.test.labels = data.test.labels
        dataTransformed.test.num_examples = data.test.num_examples
        
        return dataTransformed
    def getSingleImageFromTrainingSet(self):
        data = self.get()
        index = random.randint(0, data.train.num_examples)
        return (data.train.labels[index], data.train.images[index].reshape(28,28))