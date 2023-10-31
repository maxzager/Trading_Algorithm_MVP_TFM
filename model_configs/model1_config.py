from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import tensorflow as tf

CONFIG = {
    'model_type': 'Dense',
    'model_name': 'model1',
}

#Modelling parameters
NEURONS = 100
DROPOUT_VALUE = 0.2
ACTIVATION = tf.keras.activations.relu

def build_model(input_shape):
    model = Sequential()
    
    model.add(Dense(NEURONS,
                    kernel_initializer='he_normal',
                    input_shape=input_shape,
                    bias_initializer='zeros'))
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT_VALUE))

    model.add(Dense(2 * NEURONS, 
                    kernel_initializer='he_normal', 
                    bias_initializer='zeros'))
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT_VALUE))

    model.add(Dense(3 * NEURONS, 
                    kernel_initializer='he_normal', 
                    bias_initializer='zeros'))
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT_VALUE))

    model.add(Dense(4 * NEURONS, 
                    kernel_initializer='he_normal', 
                    bias_initializer='zeros'))
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT_VALUE))

    model.add(Dense(1, activation='sigmoid'))
    
    return model


