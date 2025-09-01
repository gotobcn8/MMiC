DIR_DEPOSITORY = 'repository'
TIME_FORMAT = "%Y%m%d_%H_%M_%S"
LOG_DIR_TIME_FORMAT = "%Y-%m-%d-%H-%M"


LOG_PATH_KEY = 'logfiles'

CLIENT_KEY = 'client'
CLIENT_ = 'client_'
SERVER_KEY = 'server'
SERVER_ = 'server_'
CLUSTER_KEY = 'cluster'
CLUSTER_ = 'cluster_'

LOG_SUFFIX = '.log'
CSV_SUFFIX = '.csv'

DEFAULT_LOG_DIR = 'logs/'

LOG_FILE_FORMAT = '({record.level_name})[{record.time:%Y-%m-%d %H:%M:%S}] {record.filename}:{record.lineno} {record.module}: {record.message}'

TEST_GENERALIZATION_RATE = 0.2

ORIGINAL = 'original_'


PREFIX_TRAIN = "train"
PREFIX_TEST = "test"
SUFFIX_NPZ = ".npz"
PICKLE = 'pickle'
PKL = 'pkl'
NUMPYZIP = 'npz'
NUMPYFILE = 'npy'
JSON = 'json'
JSON_SUFFIX = '.json'
PICKLE_SUFFIX = '.pkl'
NUMPYZIP_SUFFIX = '.npz'
NUMPYFILE_SUFFIX = '.npy'
_PICKLE = '.pickle'

DataType = ('train','test','val')
