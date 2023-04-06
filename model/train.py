#!/usr/bin/env python
#title           :train.py
#description     :to train the model
#author          :Fang Wang
#date            :2020/05/16
#usage           :python train.py --options
#python_version  :3.7.4 

from Network import Generator
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

def my_MSE_weighted(y_true, y_pred):
  weights= tf.clip_by_value(y_true, K.log(0.1+1), K.log(100.0+1))
  return K.mean(tf.multiply(weights, tf.abs(tf.subtract(y_pred, y_true))))

n_classes=4
def weighted_categorical_crossentropy(weights):
    weights=weights.reshape((1,1,1,n_classes))
    def wcce(y_true, y_pred):
        Kweights=K.constant(weights)
        y_true=K.cast(y_true,y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

class_weights=np.array([4, 19, 23, 56]) # inverse percentage of classes
class_loss = weighted_categorical_crossentropy(weights=class_weights)



#np.random.seed(100)
image_shape_hr = (96,132,1)
image_shape_lr=(8,11,1)
downscale_factor = 12
# load low resolution other variables data for training
# load low resolution data for training reforecast
PATH = '/scratch/users/nus/e0560091/data/'
reforecast_train=load(PATH+'X_train_apcp.npy')

# load high resolution data for training WRF
wrf_train=load(PATH+'wrf_train.npy')

#load low resolution data for validation
reforecast_val=load(PATH+'X_val_apcp.npy')

#load high resolution data for validation
wrf_val=load(PATH+'wrf_val.npy')

# load stage4 classified image for training
reanalysis_class_train=load(PATH+'y_class_train.npy')
reanalysis_class_train_vector=reanalysis_class_train.reshape(-1,)
reanalysis_class_val=load(PATH+'y_class_val.npy')
# convert stage4_class to categorical variable

reanalysis_class_train=to_categorical(reanalysis_class_train, num_classes=n_classes)
reanalysis_class_val=to_categorical(reanalysis_class_val, num_classes=n_classes)

#****************************************************************************************

def train(epochs, batch_size):
    
    x_train_lr=reforecast_train
    y_train_hr=wrf_train
    
    x_val_lr=reforecast_val
    y_val_hr=wrf_val  
    
    x_train_lr=reforecast_train
    y_train_class=reanalysis_class_train
    y_train_hr=wrf_train
    
    x_val_lr=reforecast_val
    y_val_class=reanalysis_class_val
    y_val_hr=wrf_val   
    
 #   loss=MSE_LOSS(image_shape_hr)
    
    batch_count = int(x_train_lr.shape[0] / batch_size)
    
    generator = Generator(image_shape_lr).generator()
    generator.compile(loss=[class_loss,my_MSE_weighted], optimizer = Adam(learning_rate=0.0001, beta_1=0.9), loss_weights=[0.01, 1.0],metrics=['mae', 'mse'])
    loss_file = open('losses.txt' , 'w+')
    loss_file.close()
        
    for e in range(1, epochs+1):
        
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        
        for _ in tqdm(range(batch_count)):
            
            rand_nums = np.random.randint(0, x_train_lr.shape[0], size=batch_size)
            
            x_lr = x_train_lr[rand_nums]
            y_hr = y_train_hr[rand_nums]
            y_class=y_train_class[rand_nums]

            gen_loss=generator.train_on_batch(x_lr, [y_class,y_hr])
        gen_loss = str(gen_loss)
        val_loss = generator.evaluate(x_val_lr, [y_val_class, y_val_hr], verbose=0)
        val_loss = str(val_loss)
        loss_file = open('losses.txt' , 'a') 
        loss_file.write('epoch%d : generator_loss = %s; validation_loss = %s\n' 
                        %(e, gen_loss, val_loss))
        
        loss_file.close()
        if e <=20:
            if e  % 5== 0:
                generator.save('gen_model%d.h5' % e)
        else:
             if e  % 10 == 0:
                generator.save('gen_model%d.h5' % e)
        


train(5, 64)


