from Network import Generator
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

# def my_MSE_weighted(y_true, y_pred):
#   weights= tf.clip_by_value(y_true, K.log(0.1+1), K.log(100.0+1))
#   return K.mean(tf.multiply(weights, tf.abs(tf.subtract(y_pred, y_true))))

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

def make_FSS_loss(mask_size):  # choose any mask size for calculating densities

    def my_FSS_loss(y_true, y_pred):

        # First: DISCRETIZE y_true and y_pred to have only binary values 0/1 
        # (or close to those for soft discretization)
        want_hard_discretization = False

        # This example assumes that y_true, y_pred have the shape (None, N, N, 1).
        
        cutoff = 0.5  # choose the cut off value for discretization

        if (want_hard_discretization):
           # Hard discretization:
           # can use that in metric, but not in loss
           y_true_binary = tf.where(y_true>cutoff, 1.0, 0.0)
           y_pred_binary = tf.where(y_pred>cutoff, 1.0, 0.0)

        else:
           # Soft discretization
           c = 10 # make sigmoid function steep
           y_true_binary = tf.math.sigmoid( c * ( y_true - cutoff ))
           y_pred_binary = tf.math.sigmoid( c * ( y_pred - cutoff ))

        # Done with discretization.

        # To calculate densities: apply average pooling to y_true.
        # Result is O(mask_size)(i,j) in Eq. (2) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (2).
        pool1 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size), strides=(1, 1), 
           padding='same')
        y_true_density = pool1(y_true_binary);
        # Need to know for normalization later how many pixels there are after pooling
        n_density_pixels = tf.cast( (tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]) , 
           tf.float32 )

        # To calculate densities: apply average pooling to y_pred.
        # Result is M(mask_size)(i,j) in Eq. (3) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (3).
        pool2 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size),
                                                 strides=(1, 1), padding='same')
        y_pred_density = pool2(y_pred_binary);

        # This calculates MSE(n) in Eq. (5) of [RL08].
        # Since we use MSE function, this automatically includes the factor 1/(Nx*Ny) in Eq. (5).
        MSE_n = tf.keras.losses.MeanSquaredError()(y_true_density, y_pred_density)

        # To calculate MSE_n_ref in Eq. (7) of [RL08] efficiently:
        # multiply each image with itself to get square terms, then sum up those terms.

        # Part 1 - calculate sum( O(n)i,j^2
        # Take y_true_densities as image and multiply image by itself.
        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        # Flatten result, to make it easier to sum over it.
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        # Calculate sum over all terms.
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        # Same for y_pred densitites:
        # Multiply image by itself
        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        # Flatten result, to make it easier to sum over it.
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        # Calculate sum over all terms.
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)
    
        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels
        
        # FSS score according to Eq. (6) of [RL08].
        # FSS = 1 - (MSE_n / MSE_n_ref)

        # FSS is a number between 0 and 1, with maximum of 1 (optimal value).
        # In loss functions: We want to MAXIMIZE FSS (best value is 1), 
        # so return only the last term to minimize.

        # Avoid division by zero if MSE_n_ref == 0
        # MSE_n_ref = 0 only if both input images contain only zeros.
        # In that case both images match exactly, i.e. we should return 0.
        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if (want_hard_discretization):
           if MSE_n_ref == 0:
              return( MSE_n )
           else:
              return( MSE_n / MSE_n_ref )
        else:
           return (MSE_n / (MSE_n_ref + my_epsilon) )

    return my_FSS_loss 

mask_size = 3 


image_shape_hr = (96,132,1)
image_shape_lr = (8,11,13) # coarse input
downscale_factor = 12
# load low resolution other variables data for training
# load low resolution data for training reforecast
PATH = '/scratch/users/nus/e0560091/data/' # CHANGE TO YOUR OWN PATH
reforecast_train=load(PATH+'X_train_ensemble.npy')

# load high resolution data for training WRF
yhr_train=load(PATH+'y_hr_train.npy')

#load low resolution data for validation
reforecast_val=load(PATH+'X_val_ensemble.npy')

#load high resolution data for validation
yhr_val=load(PATH+'y_hr_val.npy')

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
    y_train_hr=yhr_train
    
    x_val_lr=reforecast_val
    y_val_hr=yhr_val  
    
    x_train_lr=reforecast_train
    y_train_class=reanalysis_class_train
    y_train_hr=yhr_train
    
    x_val_lr=reforecast_val
    y_val_class=reanalysis_class_val
    y_val_hr=yhr_val   
    
    batch_count = int(x_train_lr.shape[0] / batch_size)
    
    generator = Generator(image_shape_lr).generator()
    generator.compile(loss=[class_loss, make_FSS_loss(mask_size)], optimizer = Adam(learning_rate=0.0001, beta_1=0.9), loss_weights=[0.01, 1.0],metrics=['mae', 'mse'])
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
        if e <=10:
            if e  % 5== 0:
                generator.save('gen_model%d.h5' % e)
        else:
             if e  % 10 == 0:
                generator.save('gen_model%d.h5' % e)
        


train(5, 64)


