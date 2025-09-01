import os


def makedirs(dir_path,mode = 755):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path,mode=mode)