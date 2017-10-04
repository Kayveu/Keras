#Need to import from individual methods rather than the entire framework, why?
#from keras.models import Sequential
import keras

#sequential is the initial buiding block of keras
#a linear pipeline of neural network layers
model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim = 8, kernel_initializer = 'random_uniform'))  #dense is not defined
model.add(keras.layers.Activation('sigmoid'))
