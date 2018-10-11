from Shared import MNIST, FMNIST, CIFAR10, CIFAR100, Utils
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras import Sequential
import types

class AutoEncoder:
    epochs = 200
    batchSize = 1000
    _dir = 'RunData'
    def __init__(self, identifier):
        self.identifier = identifier
        return;
        
    def _getEncoderModelUrl(self):
        return self._dir + "/" + self.identifier + "/encoder.h5"
    def _getDecoderModelUrl(self):
        return self._dir + "/" + self.identifier + "/decoder.h5"
    def _getAutoEncoderModelUrl(self):
        return self._dir + "/" + self.identifier + "/autoEncoder.h5"   
    def _getModelFitUrl(self):
        return self.identifier + "/modelFit.json" 
    def _getConfigurationUrl(self):
        return self.identifier + "/configuration.json" 
    
    def train(self, data):
        configuration = types.SimpleNamespace()
        configuration.optimizer = 'adam'
        configuration.loss = 'binary_crossentropy'
        configuration.epochs = self.epochs
        configuration.batchSize = self.batchSize
        configuration.trainSize = data.train.num_examples
        configuration.validationSize = data.validation.num_examples
        
        numberOfInputs = data.train.images.shape[1]
        numberEncoding = int(numberOfInputs/2)
        preNumberEncoding = 2 * int(numberOfInputs/3)
        
        autoencoder = Sequential()
        
        # Encoder Layers
        autoencoder.add(Dense(preNumberEncoding, input_shape=(numberOfInputs,), activation='relu'))
        autoencoder.add(Dense(numberEncoding, activation='relu'))
        
        # Decoder Layers
        autoencoder.add(Dense(preNumberEncoding, activation='relu'))
        autoencoder.add(Dense(numberOfInputs, activation='sigmoid'))
    
        # Extract Encoder
        input_img = Input(shape=(numberOfInputs,))
        encoder_layer1 = autoencoder.layers[0]
        encoder_layer2 = autoencoder.layers[1]
        encoder = Model(input_img, encoder_layer2(encoder_layer1(input_img)))

        # Extract Decoder     
        input_img = Input(shape=(numberEncoding,))
        decoder_layer1 = autoencoder.layers[2]
        decoder_layer2 = autoencoder.layers[3]
        decoder = Model(input_img, decoder_layer2(decoder_layer1(input_img)))
                
#        # Setup Architecture
#        input_img = Input(shape=(numberOfInputs,))
#        encoded = Dense(numberEncoding, activation='relu')(input_img)         
#        decoded = Dense(numberOfInputs, activation='sigmoid')(encoded)
#
#        # Create Models
#        autoencoder = Model(input_img, decoded)
#        encoder = Model(input_img, encoded)
#        encoded_input = Input(shape=(numberEncoding,))
#        decoder_layer = autoencoder.layers[-1]
#        decoder = Model(encoded_input, decoder_layer(encoded_input))

        # Train
        autoencoder.compile(optimizer=configuration.optimizer, loss=configuration.loss)
        modelFit = autoencoder.fit(data.train.images, data.train.images,
                        epochs=self.epochs,
                        batch_size=self.batchSize,
                        shuffle=True,
                        validation_data=(data.validation.images, data.validation.images))
        
        # Save models
        encoder.save(self._getEncoderModelUrl())
        decoder.save(self._getDecoderModelUrl())
        autoencoder.save(self._getAutoEncoderModelUrl())
        Utils.ResultsManager().save(self._getModelFitUrl(), modelFit)
        Utils.ResultsManager().save(self._getConfigurationUrl(), configuration)
        return modelFit
    
    def autoEncoderModel(self):
        return load_model(self._getAutoEncoderModelUrl())

    def encode(self, images):
        encoder = load_model(self._getEncoderModelUrl())
        return encoder.predict(images)
    
    def decode(self, encodedImages):
        decoder = load_model(self._getDecoderModelUrl())
        return decoder.predict(encodedImages)
    
    def autoEncode(self, images):
        autoencoder = load_model(self._getAutoEncoderModelUrl())
        return autoencoder.predict(images)
    
    def getModelFit(self):
        return Utils.ResultsManager().load(self._getModelFitUrl())
    
    def getConfiguration(self):
        return Utils.ResultsManager().load(self._getConfigurationUrl())
  
    def display(self, data, dim1 = 28, dim2 = 28, dim3 = 1):
        import matplotlib.pyplot as plt
        encoded = self.encode(data.test.images)
        decoded_imgs = self.decode(encoded)
        
        n = 5
        plt.figure(figsize=(10, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            if (dim3 > 1):
                plt.imshow(data.test.images[i].reshape(dim1,dim2,dim3))
            else:
                plt.imshow(data.test.images[i].reshape(dim1,dim2))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            ax = plt.subplot(2, n, i + 1 + n)
            if (dim3 > 1):
                plt.imshow(decoded_imgs[i].reshape(dim1,dim2,dim3))
            else:
                plt.imshow(decoded_imgs[i].reshape(dim1,dim2))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

def main(): 
    '''MNIST'''
    print('')
    print('Running MNIST')     
    data = MNIST.DataProvider().get()
    ae = AutoEncoder('AE_MNIST')
    ae.train(data)
    ae.display(data, 28, 28, 1)
    
    '''FMNIST'''
    print('')
    print('Running FMNIST')     
    data = FMNIST.DataProvider().get()
    ae = AutoEncoder('AE_FMNIST')
    ae.train(data)
    ae.display(data, 28, 28, 1)
    
    '''CIFAR10'''
    print('')
    print('Running CIFAR10')     
    data = CIFAR10.DataProvider().get()
    ae = AutoEncoder('AE_CIFAR10')
    ae.train(data)
    ae.display(data, 32, 32, 3)
    
    '''CIFAR100'''
    print('')
    print('Running CIFAR100')     
    data = CIFAR100.DataProvider().get()
    ae = AutoEncoder('AE_CIFAR100')
    ae.train(data)
    ae.display(data, 32, 32, 3)

if __name__ == '__main__':
    main()