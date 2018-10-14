from Shared import MNIST, FMNIST, CIFAR10, CIFAR100, Utils
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
import types
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
import pandas as pd
import matplotlib.pyplot as plt

class CNN:
    epochs = 25
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
    
    def train(self, data, shape, applyL1 = False, applyL2 = False, applyDropout = False, applyBatch = False):
        dropoutRate = 0.1
        regularizerRate = 0.01
        print(applyDropout)
        configuration = types.SimpleNamespace()
        configuration.activation = 'relu'
        configuration.optimizer = 'adam' #optimizers.SGD(lr=0.1)
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
        
        regularizer = None
        if applyL1:
            regularizer = regularizers.l1(regularizerRate)
        if applyL2:
            regularizer = regularizers.l2(regularizerRate)    
        
        model.add(Conv2D(55, filterSize, kernel_initializer=kernel_initializer, kernel_regularizer=regularizer, strides=(1, 1), padding='valid', activation=configuration.activation, input_shape=(configuration.shape1,configuration.shape2,configuration.shape3)))
        model.add(MaxPooling2D(pool_size=poolSize))
        if applyBatch:
            model.add(BatchNormalization())
        if applyDropout:
            model.add(Dropout(dropoutRate))

        model.add(Conv2D(55, filterSize, kernel_initializer=kernel_initializer, kernel_regularizer=regularizer, strides=(1, 1), padding='valid', activation=configuration.activation, input_shape=(configuration.shape1,configuration.shape2,configuration.shape3)))
        model.add(MaxPooling2D(pool_size=poolSize))
        if applyBatch:
            model.add(BatchNormalization())
        if applyDropout:
            model.add(Dropout(dropoutRate))
            
        model.add(Conv2D(55, filterSize, kernel_initializer=kernel_initializer, kernel_regularizer=regularizer, strides=(1, 1), padding='valid', activation=configuration.activation, input_shape=(configuration.shape1,configuration.shape2,configuration.shape3)))
        model.add(MaxPooling2D(pool_size=poolSize))
        if applyBatch:
            model.add(BatchNormalization())
        if applyDropout:
            model.add(Dropout(dropoutRate))

        model.add(Flatten())
        model.add(Dense(200, activation=configuration.activation, kernel_regularizer=regularizer, kernel_initializer=kernel_initializer))
        if applyBatch:
            model.add(BatchNormalization())
        if applyDropout:
            model.add(Dropout(dropoutRate))
        model.add(Dense(100, activation=configuration.activation, kernel_regularizer=regularizer, kernel_initializer=kernel_initializer))
        if applyBatch:
            model.add(BatchNormalization())
        if applyDropout:
            model.add(Dropout(dropoutRate))
        model.add(Dense(50, activation=configuration.activation, kernel_regularizer=regularizer, kernel_initializer=kernel_initializer))
        if applyBatch:
            model.add(BatchNormalization())
        if applyDropout:
            model.add(Dropout(dropoutRate))
        model.add(Dense(25, activation=configuration.activation, kernel_regularizer=regularizer, kernel_initializer=kernel_initializer))
        if applyBatch:
            model.add(BatchNormalization())
        if applyDropout:
            model.add(Dropout(dropoutRate))
        model.add(Dense(configuration.classes, activation='softmax', kernel_regularizer=regularizer, kernel_initializer=kernel_initializer))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=configuration.optimizer,
                      metrics=['accuracy'])                
        model.summary()
        modelFit = model.fit(data.train.images, data.train.labels,
                  batch_size=self.batchSize,
                  epochs=self.epochs,
                  shuffle=True,
                  validation_data=(data.validation.images, data.validation.labels))

        testAccuracy = model.evaluate(data.test.images, data.test.labels)
        model.save(self._getModelUrl())
        Utils.ResultsManager().save(self._getModelFitUrl(), modelFit)
        Utils.ResultsManager().save(self._getTestAccuracyUrl(), testAccuracy)
#        Utils.ResultsManager().save(self._getConfigurationUrl(), configuration)
        return 
    
    def model(self):
        return load_model(self._getModelUrl())

    def getModelFit(self):
        return Utils.ResultsManager().load(self._getModelFitUrl())
    
    def getConfiguration(self):
        return Utils.ResultsManager().load(self._getConfigurationUrl())
    
    def getTestAccuracy(self):
        return Utils.ResultsManager().load(self._getTestAccuracyUrl())
#
#'''MNIST'''
#print('')
#print('Running MNIST')    
#data = MNIST.DataProvider().get()
#shape = (28, 28, 1)
#data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
#data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
#data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
#cnn = CNN('CNN_MNIST')
#cnn.train(data, shape)
#cnn = CNN('CNN_MNIST_L1')
#cnn.train(data, shape, True, False, False, False)
#cnn = CNN('CNN_MNIST_L2')
#cnn.train(data, shape, False, True, False, False)
#cnn = CNN('CNN_MNIST_D')
#cnn.train(data, shape, False, False, True, False)
#cnn = CNN('CNN_MNIST_B')
#cnn.train(data, shape, False, False, False, True)
#
#'''FMNIST'''
#print('')
#print('Running FMNIST')    
#data = FMNIST.DataProvider().get()
#shape = (28, 28, 1)
#data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
#data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
#data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
#cnn = CNN('CNN_FMNIST')
#cnn.train(data, shape)
#cnn = CNN('CNN_FMNIST_L1')
#cnn.train(data, shape, True, False, False, False)
#cnn = CNN('CNN_FMNIST_L2')
#cnn.train(data, shape, False, True, False, False)
#cnn = CNN('CNN_FMNIST_D')
#cnn.train(data, shape, False, False, True, False)
#cnn = CNN('CNN_FMNIST_B')
#cnn.train(data, shape, False, False, False, True)
#
#'''CIFAR10'''
#print('')
#print('Running CIFAR10')    
#data = CIFAR10.DataProvider().get()
#shape = (32, 32, 3)
#data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
#data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
#data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
#cnn = CNN('CNN_CIFAR10')
#cnn.train(data, shape)
#cnn = CNN('CNN_CIFAR10_L1')
#cnn.train(data, shape, True, False, False, False)
#cnn = CNN('CNN_CIFAR10_L2')
#cnn.train(data, shape, False, True, False, False)
#cnn = CNN('CNN_CIFAR10_D')
#cnn.train(data, shape, False, False, True, False)
#cnn = CNN('CNN_CIFAR10_B')
#cnn.train(data, shape, False, False, False, True)
#
#'''CIFAR100'''
#print('')
#print('Running CIFAR100')    
#data = CIFAR100.DataProvider().get()
#shape = (32, 32, 3)
#data.train.images = data.train.images.reshape(data.train.images.shape[0], shape[0], shape[1], shape[2])
#data.validation.images = data.validation.images.reshape(data.validation.images.shape[0], shape[0], shape[1], shape[2])
#data.test.images = data.test.images.reshape(data.test.images.shape[0], shape[0], shape[1], shape[2])
#cnn = CNN('CNN_CIFAR100')
#cnn.train(data, shape)
#cnn = CNN('CNN_CIFAR100_L1')
#cnn.train(data, shape, True, False, False, False)
#cnn = CNN('CNN_CIFAR100_L2')
#cnn.train(data, shape, False, True, False, False)
#cnn = CNN('CNN_CIFAR100_D')
#cnn.train(data, shape, False, False, True, False)
#cnn = CNN('CNN_CIFAR100_B')
#cnn.train(data, shape, False, False, False, True)

def plotModelFit(modelFit,testAccuracy,textTop,ylim=(0,100),save1='',save2=''):
    df = pd.DataFrame({
        'Training Loss':modelFit.history['loss'],
        'Validation Loss':modelFit.history['val_loss']
        },index=range(len(modelFit.history['val_loss'])))
    
    plt.style.use('seaborn-darkgrid')
    plt.plot(df.index, df['Training Loss'], marker='', color='#770000', label='Training')
    plt.plot(df.index, df['Validation Loss'], marker='', color='#444444', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Cross Entropy)')
    plt.legend(loc='upper right')
    plt.savefig('Results/' + save1 + '.png')   
    plt.close()
    
    df = pd.DataFrame({
        'Training Accuracy':modelFit.history['acc'],
        'Validation Accuracy':modelFit.history['val_acc']
        },index=range(len(modelFit.history['val_acc'])))
        
    plt.style.use('seaborn-darkgrid')
    plt.plot(df.index, df['Training Accuracy']*100, marker='', color='#770000', label='Training')
    plt.plot(df.index, df['Validation Accuracy']*100, marker='', color='#444444', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(ylim[0],ylim[1])
    plt.legend()
    plt.plot(len(df)-1,testAccuracy*100,'X',color='#008811',markersize=14) 
    plt.text(10,textTop,'Test Accuracy: {0:.2f}%'.format(testAccuracy*100), horizontalalignment='left', size='small', color='#222222')
    plt.savefig('Results/' + save2 + '.png')   
    plt.close()

print('MNIST')
testAccuracies = []
cnn = CNN('CNN_MNIST')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,97,(95,100),'M4_MNIST_LOSS','M4_MNIST_ACC')
cnn = CNN('CNN_MNIST_L1')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,11,(10.5,12),'M4_MNIST_L1_LOSS','M4_MNIST_L1_ACC')
cnn = CNN('CNN_MNIST_L2')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,94,(92,100),'M4_MNIST_L2_LOSS','M4_MNIST_L2_ACC')
cnn = CNN('CNN_MNIST_D')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,98,(97,100),'M4_MNIST_D_LOSS','M4_MNIST_D_ACC')
cnn = CNN('CNN_MNIST_B')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,98,(97,100),'M4_MNIST_B_LOSS','M4_MNIST_B_ACC')
df = pd.DataFrame({
    'Base':testAccuracies[0],
    'L1':testAccuracies[1],
    'L2':testAccuracies[2],
    'Dropout':testAccuracies[3],
    'Batch':testAccuracies[4]
    },index=range(1))*100

print('FMNIST')
testAccuracies = []
cnn = CNN('CNN_FMNIST')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,83,(80,100),'M4_FMNIST_LOSS','M4_FMNIST_ACC')
cnn = CNN('CNN_FMNIST_L1')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,11,(8,12),'M4_FMNIST_L1_LOSS','M4_FMNIST_L1_ACC')
cnn = CNN('CNN_FMNIST_L2')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,72.5,(65,85),'M4_FMNIST_L2_LOSS','M4_FMNIST_L2_ACC')
cnn = CNN('CNN_FMNIST_D')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,87,(85,92),'M4_FMNIST_D_LOSS','M4_FMNIST_D_ACC')
cnn = CNN('CNN_FMNIST_B')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,82,(80,100),'M4_FMNIST_B_LOSS','M4_FMNIST_B_ACC')
df = df.append(pd.DataFrame({
    'Base':testAccuracies[0],
    'L1':testAccuracies[1],
    'L2':testAccuracies[2],
    'Dropout':testAccuracies[3],
    'Batch':testAccuracies[4]
    },index=range(1))*100)

print('CIFAR10')
testAccuracies = []
cnn = CNN('CNN_CIFAR10')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,60,(50,90),'M4_CIFAR10_LOSS','M4_CIFAR10_ACC')
cnn = CNN('CNN_CIFAR10_L1')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,11,(8,12),'M4_CIFAR10_L1_LOSS','M4_CIFAR10_L1_ACC')
cnn = CNN('CNN_CIFAR10_L2')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,20,(10,55),'M4_CIFAR10_L2_LOSS','M4_CIFAR10_L2_ACC')
cnn = CNN('CNN_CIFAR10_D')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,60,(50,85),'M4_CIFAR10_D_LOSS','M4_CIFAR10_D_ACC')
cnn = CNN('CNN_CIFAR10_B')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,60,(50,95),'M4_CIFAR10_B_LOSS','M4_CIFAR10_B_ACC')
df = df.append(pd.DataFrame({
    'Base':testAccuracies[0],
    'L1':testAccuracies[1],
    'L2':testAccuracies[2],
    'Dropout':testAccuracies[3],
    'Batch':testAccuracies[4]
    },index=range(1))*100)

print('CIFAR100')
testAccuracies = []
cnn = CNN('CNN_CIFAR100')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,22,(15,50),'M4_CIFAR100_LOSS','M4_CIFAR100_ACC')
cnn = CNN('CNN_CIFAR100_L1')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,1.1,(0.5,1.25),'M4_CIFAR100_L1_LOSS','M4_CIFAR100_L1_ACC')
cnn = CNN('CNN_CIFAR100_L2')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,1.1,(0.5,1.25),'M4_CIFAR100_L2_LOSS','M4_CIFAR100_L2_ACC')
cnn = CNN('CNN_CIFAR100_D')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,15,(10,35),'M4_CIFAR100_D_LOSS','M4_CIFAR100_D_ACC')
cnn = CNN('CNN_CIFAR100_B')
testAccuracy = cnn.getTestAccuracy()[1]
testAccuracies.append(testAccuracy)
modelFit = cnn.getModelFit()
plotModelFit(modelFit,testAccuracy,20,(10,70),'M4_CIFAR100_B_LOSS','M4_CIFAR100_B_ACC')
df = df.append(pd.DataFrame({
    'Base':testAccuracies[0],
    'L1':testAccuracies[1],
    'L2':testAccuracies[2],
    'Dropout':testAccuracies[3],
    'Batch':testAccuracies[4]
    },index=range(1))*100)

