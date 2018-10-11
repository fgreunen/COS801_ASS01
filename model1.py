from Shared import MNIST, FMNIST, CIFAR10, CIFAR100, Utils
import tensorflow as tf
import sys 
import matplotlib.pyplot as plt
import time
import types
import numpy as np

print("python {}".format(sys.version))
print("tensorflow version {}".format(tf.__version__))

configuration = types.SimpleNamespace()
configuration.stoppingThresholdWindowSize = 15
configuration.batchSize = 50
configuration.trainingBatchesPerValidationBatch = 50
configuration.maxNumberOfTrainingSteps = 100000
configuration.learningRateDecay = 0.65

configuration.numberOfLearningRates = 4
configuration.learningRateInitial = 1
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
        initializer = tf.glorot_normal_initializer()
        W = tf.Variable(initializer([configuration.numberOfPixelsPerImage,configuration.numberOfClassLabels]))
        b = tf.Variable(tf.to_float(np.repeat(0.1, configuration.numberOfClassLabels)))
        y = tf.matmul(x,W) + b
        y_true = tf.placeholder(tf.float32,[None,configuration.numberOfClassLabels])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_true,1)),tf.float32))
        init = tf.global_variables_initializer()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=configuration.learningRates[i])
        train = optimizer.minimize(cross_entropy)
        start = time.time()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)

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


'''MNIST'''
print('')
print('Running MNIST')
dataProvider = MNIST.DataProvider()
data = dataProvider.get()
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
plt.imshow(dataProvider.getSingleImageFromTrainingSet()[1], cmap="gist_gray")
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m1.mnist.json', runs)
runs = Utils.ResultsManager().load('m1.mnist.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)

'''FMNIST'''
print('')
print('Running FMNIST')
dataProvider = FMNIST.DataProvider()
data = dataProvider.get()
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
plt.imshow(dataProvider.getSingleImageFromTrainingSet()[1], cmap="gist_gray")
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m1.fmnist.json', runs)
runs = Utils.ResultsManager().load('m1.fmnist.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)

'''CIFAR10'''
print('')
print('Running CIFAR10')
dataProvider = CIFAR10.DataProvider()
data = dataProvider.get()
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
plt.imshow(dataProvider.getSingleImageFromTrainingSet()[1])
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m1.cifar10.json', runs)
runs = Utils.ResultsManager().load('m1.cifar10.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)

'''CIFAR100'''
print('')
print('Running CIFAR100')
dataProvider = CIFAR100.DataProvider()
data = dataProvider.get()
configuration.numberOfPixelsPerImage = data.train.images.shape[1]
configuration.numberOfClassLabels = data.train.labels.shape[1]
plt.imshow(dataProvider.getSingleImageFromTrainingSet()[1])
runs = createAndRunModel(data, configuration)
Utils.ResultsManager().save('m1.cifar100.json', runs)
runs = Utils.ResultsManager().load('m1.cifar100.json')
plt.plot(range(len(runs[0].validationAccuracies)), runs[0].validationAccuracies)