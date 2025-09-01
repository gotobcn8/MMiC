import pickle
import const.constants as const
from fedlog.logbooker import glogger
import os
import numpy as np

def save_as_type(data,file_name:str,data_path:str,mode:str = 'wb', save_type:str = const.PICKLE):
    if data is None:
        glogger.error('save data is None!')
        return
    save_type = save_type.lower()
    if save_type == const.PICKLE or save_type == const.PKL:
        save_as_pickle(data,os.path.join(data_path,file_name + const.PICKLE_SUFFIX),mode)

    elif save_type == const.NUMPYFILE:
        save_as_numpy(data,os.path.join(data_path,file_name + const.NUMPYFILE_SUFFIX),mode)
        
    elif save_type == const.NUMPYZIP:
        save_as_numpyz(data,os.path.join(data_path,file_name + const.NUMPYZIP_SUFFIX),mode) 
    
def save_as_pickle(data,data_path,mode):
    with open(data_path,mode=mode) as f:
        pickle.dump(data,f)

def save_as_numpy(data,data_path,mode):
    with open(data_path,mode=mode) as f:
        np.save(f,data)
        
def save_as_numpyz(data,data_path,mode):
    with open(data_path,mode=mode) as f:
        np.savez(f,data)