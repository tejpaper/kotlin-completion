import os
import sys

RANDOM_SEED = 0
MASK_TOKEN = '<|mask|>'
PAST_CTX_LEN = 256
FUTURE_CTX_LEN = 255

CACHE_DIR = 'cache'
DATA_DIR = 'data'
MODEL_DIR = 'model'

if 'google.colab' in sys.modules:
    path2drive = '/content/drive/MyDrive'
    CACHE_DIR = os.path.join(path2drive, CACHE_DIR)
    DATA_DIR = os.path.join(path2drive, DATA_DIR)
    MODEL_DIR = os.path.join(path2drive, MODEL_DIR)

KOTLIN_TRAIN_CSV = os.path.join(DATA_DIR, 'kotlin', 'train.csv')
KOTLIN_DEV_CSV = os.path.join(DATA_DIR, 'kotlin', 'dev.csv')
KOTLIN_TEST_CSV = os.path.join(DATA_DIR, 'kotlin', 'test.csv')
PYTHON_TEST_CSV = os.path.join(DATA_DIR, 'python', 'test.csv')

KOTLIN_BASELINE = os.path.join(CACHE_DIR, 'kotlin-baseline.pt')
KOTLIN_TUNED = os.path.join(CACHE_DIR, 'kotlin-tuned.pt')
PYTHON_BASELINE = os.path.join(CACHE_DIR, 'python-baseline.pt')
PYTHON_TUNED = os.path.join(CACHE_DIR, 'python-tuned.pt')

