from functools import partial
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler


def timecheck(train, val, test):
    '''
    Return missing time step in train, val and test split
    '''
    train_timecheck = np.arange(np.datetime64("2000-01-01T06"), np.datetime64("2014-01-01T12"), np.timedelta64(6, "h"))
    val_timecheck = np.arange(np.datetime64("2014-01-01T12"), np.datetime64("2016-12-31T18"), np.timedelta64(6, "h"))
    test_timecheck = np.arange(np.datetime64("2016-12-31T18"), np.datetime64("2020-01-01T00"), np.timedelta64(6, "h"))
    print('train:', set(train_timecheck) - set(train.time.values.astype('datetime64[h]')))
    print('val:', set(val_timecheck) - set(val.time.values.astype('datetime64[h]')))
    print('test:', set(test_timecheck) - set(test.time.values.astype('datetime64[h]')))
    return


def trainvaltest(dataset):
    '''
    Return train, val, test dataset
    '''
    train = dataset.sel(time=slice('2000-01-01T06', '2014-01-01T06'))
    val = dataset.sel(time=slice('2014-01-01T12', '2016-12-31T12'))
    test = dataset.sel(time=slice('2016-12-31T18', '2019-12-31T18'))
    return train, val, test


def resampling(partial_dataset, var):
    partial_dataset = partial_dataset.resample(time='6H').asfreq()
    if var == 'cape':
        partial_dataset.cape.values = partial_dataset.cape.interpolate_na(dim='time')
        return partial_dataset
    if var == 'pwat':
        partial_dataset.pwat.values = partial_dataset.pwat.interpolate_na(dim='time')
        return partial_dataset
    if var == 'apcp':
        partial_dataset.tp.values = partial_dataset.tp.interpolate_na(dim='time')
        return partial_dataset
    if var == 't2m':
        partial_dataset.t2m.values = partial_dataset.t2m.interpolate_na(dim='time')
        return partial_dataset
    if var == 'cin':
        partial_dataset.cin.values = partial_dataset.cin.interpolate_na(dim='time')
        return partial_dataset    
    else:
        print('Error: var not found in function, please define')

def transform_train(trainds, var):
    if var == 'apcp': # tp
        scaler_train = MinMaxScaler()
        X_one_col = trainds.tp.values.reshape([trainds.tp.values.shape[0]*trainds.tp.values.shape[1]*trainds.tp.values.shape[2], 1])
        X_one_col = np.log10(X_one_col+1) # 10**X_one_col - 1 to scale back
        X_one_col_res = scaler_train.fit_transform(X_one_col) # scaler_train.inverse_transform(X_one_col_res) to scale back, or use 10**scaler_train.inverse_transform(X_one_col_res) -1 only
        trainds.tp.values = X_one_col_res.reshape(trainds.tp.values.shape)
        return scaler_train, trainds

    if var == 'RAINNC': # tp
        scaler_train = MinMaxScaler()
        X_one_col = trainds.RAINNC.values.reshape([trainds.RAINNC.values.shape[0]*trainds.RAINNC.values.shape[1]*trainds.RAINNC.values.shape[2], 1])
        X_one_col = np.log10(X_one_col+1) # 10**X_one_col - 1 to scale back
        X_one_col_res = scaler_train.fit_transform(X_one_col) # scaler_train.inverse_transform(X_one_col_res) to scale back, or use 10**scaler_train.inverse_transform(X_one_col_res) -1 only
        trainds.RAINNC.values = X_one_col_res.reshape(trainds.RAINNC.values.shape)
        return scaler_train, trainds

    if var == 'pwat':
        scaler_train = MinMaxScaler()
        X_one_col = trainds.pwat.values.reshape([trainds.pwat.values.shape[0]*trainds.pwat.values.shape[1]*trainds.pwat.values.shape[2], 1])
        # X_one_col = np.log10(X_one_col+1) # 10**X_one_col - 1 to scale back
        X_one_col_res = scaler_train.fit_transform(X_one_col) # scaler_train.inverse_transform(X_one_col_res) to scale back, or use 10**scaler_train.inverse_transform(X_one_col_res) -1 only
        trainds.pwat.values = X_one_col_res.reshape(trainds.pwat.values.shape)
        return scaler_train, trainds

    if var == 'cape':
        scaler_train = MinMaxScaler()
        X_one_col = trainds.cape.values.reshape([trainds.cape.values.shape[0]*trainds.cape.values.shape[1]*trainds.cape.values.shape[2], 1])
        # X_one_col = np.log10(X_one_col+1) # 10**X_one_col - 1 to scale back
        X_one_col_res = scaler_train.fit_transform(X_one_col) # scaler_train.inverse_transform(X_one_col_res) to scale back, or use 10**scaler_train.inverse_transform(X_one_col_res) -1 only
        trainds.cape.values = X_one_col_res.reshape(trainds.cape.values.shape)
        return scaler_train, trainds

    if var == 'cin':
        scaler_train = MinMaxScaler()
        X_one_col = trainds.cin.values.reshape([trainds.cin.values.shape[0]*trainds.cin.values.shape[1]*trainds.cin.values.shape[2], 1])
        # X_one_col = np.log10(X_one_col+1) # 10**X_one_col - 1 to scale back
        X_one_col_res = scaler_train.fit_transform(X_one_col) # scaler_train.inverse_transform(X_one_col_res) to scale back, or use 10**scaler_train.inverse_transform(X_one_col_res) -1 only
        trainds.cin.values = X_one_col_res.reshape(trainds.cin.values.shape)
        return scaler_train, trainds

    if var == 't2m':
        scaler_train = MinMaxScaler()
        X_one_col = trainds.t2m.values.reshape([trainds.t2m.values.shape[0]*trainds.t2m.values.shape[1]*trainds.t2m.values.shape[2], 1])
        # X_one_col = np.log10(X_one_col+1) # 10**X_one_col - 1 to scale back
        X_one_col_res = scaler_train.fit_transform(X_one_col) # scaler_train.inverse_transform(X_one_col_res) to scale back, or use 10**scaler_train.inverse_transform(X_one_col_res) -1 only
        trainds.t2m.values = X_one_col_res.reshape(trainds.t2m.values.shape)
        return scaler_train, trainds
    
    else:
        print('Error: var not found in function, please define')


def transform_val_test(val_test, scaler_train, is_prec=True):
    '''
    Input (example): ds_val_apcp.tp, scaler_train, True/False
    Output: Transformed validation/test XR data
    If is_prec set to True, variable is precipitation
    '''
    if is_prec == True:
        X_one_col = val_test.values.reshape([val_test.values.shape[0]*val_test.values.shape[1]*val_test.values.shape[2], 1])
        X_one_col = np.log10(X_one_col+1) 
        X_one_col_res = scaler_train.transform(X_one_col) 
        val_test.values = X_one_col_res.reshape(val_test.values.shape)
        return val_test.values
        
    else:
        X_one_col = val_test.values.reshape([val_test.values.shape[0]*val_test.values.shape[1]*val_test.values.shape[2], 1])
        # X_one_col = np.log10(X_one_col+1) 
        X_one_col_res = scaler_train.transform(X_one_col) 
        val_test.values = X_one_col_res.reshape(val_test.values.shape)
        return val_test.values
    

def inverse_val_test(transformed_vt, scaler_train, is_prec=True):
    '''
    Input (example): ds_val_apcp.tp, scaler_train, True/False
    Output: Inversed of transformed validation/test XR data
    If is_prec set to True, variable is precipitation
    '''
    if is_prec == True:
        X_one_col = transformed_vt.values.reshape([transformed_vt.values.shape[0]*transformed_vt.values.shape[1]*transformed_vt.values.shape[2], 1])
        X_one_col_res = 10**scaler_train.inverse_transform(X_one_col) -1
        transformed_vt.values = X_one_col_res.reshape(transformed_vt.values.shape)
        return transformed_vt.values
    
    else:
        X_one_col = transformed_vt.values.reshape([transformed_vt.values.shape[0]*transformed_vt.values.shape[1]*transformed_vt.values.shape[2], 1])
        X_one_col_res = scaler_train.inverse_transform(X_one_col)
        transformed_vt.values = X_one_col_res.reshape(transformed_vt.values.shape)
        return transformed_vt.values


