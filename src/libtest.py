# Modules
from tensorflow import keras
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
from keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Softmax
#from tensorflow.keras.layers import AveragePooling2D
#import tensorflow_addons as tfa

# from Network import Generator
# from MSE_LOSS import MSE_LOSS
from keras.models import Model
from keras.layers import Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
from numpy import save
from numpy import load
import tensorflow.keras.backend as K
import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
from tensorflow.keras.utils import to_categorical

print('done')