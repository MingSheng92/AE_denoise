from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

# custom loss function to measure image noise ratio 
def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    #tf.math.log
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def createModel(img_size):
    input_layer = Input(shape=(img_size[0], img_size[1], 1))

    x = Conv2D(64, (5, 5), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(input_shape=(img_size[0], img_size[1], 1))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(input_shape=(img_size[0], img_size[1], 1))(x)

    x = Dropout(0.4)(x)

    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(input_shape=(img_size[0], img_size[1], 1))(x)

    x = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(input_shape=(img_size[0], img_size[1], 1))(x)

    output = Conv2D(1, (1, 1), activation='hard_sigmoid', padding='same')(x)

    NN_model = Model(input_layer, output)
    adam = Adam(lr=1e-3)
    NN_model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
    
    return NN_model