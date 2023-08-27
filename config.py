import os

basedir = os.path.abspath(os.path.dirname(__file__))


class FlaskConfig(object):
    DEBUG = True
    
class MLConfig:
    MAX_SEQ_LENGHT = 1000
    EMBEDDING_DIM = 300
    MODEL_PATH = f'{basedir}/ml/model/model_files/main model.pt'

    MODEL_PARAMS = {
        # 'embedding_matrix'  : None, 
        'hidden_dim'        : 200, 
        'num_layers'        : 4, 
        'output_labels'     : 1,
    }