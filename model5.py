from Shared import MNIST, FMNIST, CIFAR10, CIFAR100, Utils
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D
import types

class CNN:
    epochs = 150
    batchSize = 50
    _dir = 'RunData'
    def __init__(self, identifier):
        self.identifier = identifier
        return;
        
    def _getModelUrl(self):
        return self._dir + "/" + self.identifier + "/model.h5"
    def _getModelFitUrl(self):
        return self.identifier + "/modelFit.json" 
    def _getConfigurationUrl(self):
        return self.identifier + "/configuration.json" 
    def _getTestAccuracyUrl(self):
        return self.identifier + "/testAccuracy.json" 
    
    def train(self, data, shape):
        configuration = types.SimpleNamespace()
        configuration.activation = 'relu'
        configuration.optimizer = 'adam'
        configuration.trainSize = data.train.num_examples
        configuration.validationSize = data.validation.num_examples
        configuration.classes = data.train.labels.shape[1]
        configuration.shape1 = shape[0]
        configuration.shape2 = shape[1]
        configuration.shape3 = shape[2]
        
        filterSize = (3,3)
        poolSize = (2,2)
        model = Sequential()
        kernel_initializer = keras.initializers.glorot_normal(seed=None)
        model.add(Conv2D(55, filterSize, kernel_initializer=kernel_initializer, strides=(1, 1), padding='valid', activation=configuration.activation, input_shape=(configuration.shape1,configuration.shape2,configuration.shape3)))
        model.add(MaxPooling2D(pool_size=poolSize))
        model.add(Conv2D(55, filterSize, kernel_initializer=kernel_initializer, strides=(1, 1), padding='valid', activation=configuration.activation, input_shape=(configuration.shape1,configuration.shape2,configuration.shape3)))
        model.add(MaxPooling2D(pool_size=poolSize))
        model.add(Conv2D(55, filterSize, kernel_initializer=kernel_initializer, strides=(1, 1), padding='valid', activation=configuration.activation, input_shape=(configuration.shape1,configuration.shape2,configuration.shape3)))
        model.add(MaxPooling2D(pool_size=poolSize))

        model.add(Flatten())
        model.add(Dense(200, activation=configuration.activation, kernel_initializer=kernel_initializer))
        model.add(Dense(100, activation=configuration.activation, kernel_initializer=kernel_initializer))
        model.add(Dense(50, activation=configuration.activation, kernel_initializer=kernel_initializer))
        model.add(Dense(25, activation=configuration.activation, kernel_initializer=kernel_initializer))
        model.add(Dense(configuration.classes, activation='softmax', kernel_initializer=kernel_initializer))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=configuration.optimizer,
                      metrics=['accuracy'])                

        modelFit = model.fit(data.train.images, data.train.labels,
                  batch_size=self.batchSize,
                  epochs=self.epochs,
                  shuffle=True,
                  validation_data=(data.validation.images, data.validation.labels))

        testAccuracy = model.evaluate(data.test.images, data.test.labels)
        model.save(self._getModelUrl())
        Utils.ResultsManager().save(self._getModelFitUrl(), modelFit)
        Utils.ResultsManager().save(self._getConfigurationUrl(), configuration)
        Utils.ResultsManager().save(self._getTestAccuracyUrl(), testAccuracy)
        return modelFit, testAccuracy
    
    def model(self):
        return load_model(self._getModelUrl())

    def getModelFit(self):
        return Utils.ResultsManager().load(self._getModelFitUrl())
    
    def getConfiguration(self):
        return Utils.ResultsManager().load(self._getConfigurationUrl())
    
    def getTestAccuracy(self):
        return Utils.ResultsManager().load(self._getTestAccuracyUrl())

'''MNIST'''
print('')
print('Running MNIST')    
data = MNIST.DataProvider().get()
shape = (28, 28, 1)
data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
cnn = CNN('CNN_MNIST')
cnn.train(data, shape)
cnn.getTestAccuracy()

'''FMNIST'''
print('')
print('Running FMNIST')    
data = FMNIST.DataProvider().get()
shape = (28, 28, 1)
data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
cnn = CNN('CNN_FMNIST')
cnn.train(data, shape)
cnn.getTestAccuracy()

'''CIFAR10'''
print('')
print('Running CIFAR10')    
data = CIFAR10.DataProvider().get()
shape = (32, 32, 3)
data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
cnn = CNN('CNN_CIFAR10')
cnn.train(data, shape)
cnn.getTestAccuracy()

'''CIFAR100'''
print('')
print('Running CIFAR100')    
data = CIFAR100.DataProvider().get()
shape = (32, 32, 3)
data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
cnn = CNN('CNN_CIFAR100')
cnn.train(data, shape)
cnn.getTestAccuracy()

