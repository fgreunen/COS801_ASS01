from model3 import AutoEncoder
from Shared import MNIST, FMNIST, CIFAR10, CIFAR100, Utils
import tensorflow as tf
import sys 
import matplotlib.pyplot as plt
import time
import types
import numpy as np
from sklearn.preprocessing import minmax_scale

print("python {}".format(sys.version))
print("tensorflow version {}".format(tf.__version__))

configuration = types.SimpleNamespace()
configuration.stoppingThresholdWindowSize = 20
configuration.batchSize = 50
configuration.trainingBatchesPerValidationBatch = 250
configuration.maxNumberOfTrainingSteps = 100000
configuration.learningRateDecay = 0.65

configuration.numberOfLearningRates = 2
configuration.learningRateInitial = 0.3
configuration.learningRates = []
current = configuration.learningRateInitial
for i in range(configuration.numberOfLearningRates):
    configuration.learningRates.append(current)
    current = current * configuration.learningRateDecay
    
def createAndRunModel(data, configuration):
    runs = []

    for i in range(len(configuration.learningRates)): 
        print('Iteration %s of %s' %(i + 1, len(configuration.learningRates)))
        validationAccuracies = []
        trainingAccuracies = []
        x = tf.placeholder(tf.float32, shape=[None,configuration.numberOfPixelsPerImage])    
        initializer = tf.glorot_uniform_initializer()
        
        W0 = tf.Variable(initializer([configuration.numberOfPixelsPerImage, 200]))
        b0 = tf.Variable(tf.to_float(np.repeat(0.1, 200)))
        W1 = tf.Variable(initializer([200, 100]))
        b1 = tf.Variable(tf.to_float(np.repeat(0.1, 100)))
        W2 = tf.Variable(initializer([100, 50]))
        b2 = tf.Variable(tf.to_float(np.repeat(0.1, 50)))
        W3 = tf.Variable(initializer([50, 25]))
        b3 = tf.Variable(tf.to_float(np.repeat(0.1, 25)))
        W4 = tf.Variable(initializer([25, configuration.numberOfClassLabels]))
        b4 = tf.Variable(tf.to_float(np.repeat(0.1, configuration.numberOfClassLabels))   )
    
        out0 = configuration.activationFunction(tf.matmul(x, W0) + b0)
        out1 = configuration.activationFunction(tf.matmul(out0, W1) + b1)
        out2 = configuration.activationFunction(tf.matmul(out1, W2) + b2)
        out3 = configuration.activationFunction(tf.matmul(out2, W3) + b3)
        y = configuration.activationFunction(tf.matmul(out3, W4) + b4)
        y_true = tf.placeholder(tf.float64,[None,configuration.numberOfClassLabels])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_true,1)),tf.float64))
        init = tf.global_variables_initializer()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=configuration.learningRates[i])
        
        train = optimizer.minimize(cross_entropy)
        start = time.time()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            print(sess.run(init))

            for step in range(configuration.maxNumberOfTrainingSteps):
                if (step % 200 == 0 and len(validationAccuracies) > 0):
                    percent = 100 * step / configuration.maxNumberOfTrainingSteps
                    print("\t %s, %s" %(percent, validationAccuracies[-1:][0]))

                # TRAINING
                batch_x, batch_y = data.train.next_batch(configuration.batchSize)
                sess.run(train, feed_dict={x:batch_x,y_true:batch_y}) 
                
                if (step % configuration.trainingBatchesPerValidationBatch == 0): # VALIDATION
                    validationAccuracies.append(sess.run(acc,feed_dict={x:data.validation.images,y_true:data.validation.labels}))
                    trainingAccuracies.append(sess.run(acc,feed_dict={x:batch_x,y_true:batch_y}))
                    if (len(validationAccuracies) >= configuration.stoppingThresholdWindowSize):
                        if (validationAccuracies[-configuration.stoppingThresholdWindowSize:][0] == max(validationAccuracies[-configuration.stoppingThresholdWindowSize:])):
                            break;
        
            trainingAccuracy = sess.run(acc,feed_dict={x:data.train.images,y_true:data.train.labels})
            testAccuracy = sess.run(acc,feed_dict={x:data.test.images,y_true:data.test.labels})
        end = time.time()
        elapsed = end - start
        
        run = types.SimpleNamespace()
        run.elapsed = elapsed
        run.learningRate = configuration.learningRates[i]
        run.step = step + 1
        run.trainingAccuracy = trainingAccuracy
        run.validationAccuracies = validationAccuracies
        run.testAccuracy = testAccuracy
        run.trainingAccuracies = trainingAccuracies
        runs.append(run)
    return runs

def transformData(data, ae):
    # Encode
    data.train.images = ae.encode(data.train.images)
    data.validation.images = ae.encode(data.validation.images)
    data.test.images = ae.encode(data.test.images)
    
    # Normalize
    for i in range(data.train.num_examples):
        data.train.images[i] = minmax_scale(data.train.images[i])
    for i in range(data.validation.num_examples):
        data.validation.images[i] = minmax_scale(data.validation.images[i])
    for i in range(data.test.num_examples):
        data.test.images[i] = minmax_scale(data.test.images[i])
    return data

'''MNIST'''
print('')
print('Running MNIST (relu)')
dataProvider = MNIST.DataProvider()
data = dataProvider.get()
ae = AutoEncoder('AE_MNIST')
print('Transforming...')
data = transformData(data, ae)
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
configuration.activationFunction = tf.nn.relu
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m4.mnist.relu.json', runs)
runs = Utils.ResultsManager().load('m4.mnist.relu.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)

'''FMNIST'''
print('')
print('Running FMNIST (relu)')
dataProvider = FMNIST.DataProvider()
data = dataProvider.get()
ae = AutoEncoder('AE_FMNIST')
print('Transforming...')
data = transformData(data, ae)
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
configuration.activationFunction = tf.nn.relu
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m4.fmnist.relu.json', runs)
runs = Utils.ResultsManager().load('m4.fmnist.relu.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)

'''CIFAR10'''
print('')
print('Running CIFAR10 (relu)')
dataProvider = CIFAR10.DataProvider()
data = dataProvider.get()
ae = AutoEncoder('AE_CIFAR10')
print('Transforming...')
data = transformData(data, ae)
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
configuration.activationFunction = tf.nn.relu
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m4.cifar10.relu.json', runs)
runs = Utils.ResultsManager().load('m4.cifar10.relu.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)

'''CIFAR100'''
print('')
print('Running CIFAR100 (relu)')
dataProvider = CIFAR100.DataProvider()
data = dataProvider.get()
ae = AutoEncoder('AE_CIFAR100')
print('Transforming...')
data = transformData(data, ae)
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
configuration.activationFunction = tf.nn.relu
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m4.cifar100.relu.json', runs)
runs = Utils.ResultsManager().load('m4.cifar100.relu.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)


