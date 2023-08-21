import os

basedir = os.path.abspath(os.path.dirname(__file__))


class FlaskConfig(object):
    DEBUG = True
    
class MLConfig:
    MAX_SEQ_LENGHT = 1000
    EMBEDDING_DIM = 300
    MODEL_PATH = f'{basedir}/model/test.pt'