from urllib import request

def get_remote(url,save_path = 'repository'):
    '''
    url is a link to the target file, save_path is local directory
    '''
    request.urlretrieve(url,save_path)