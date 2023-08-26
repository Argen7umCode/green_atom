from flask import Flask
from ml.dataprocess import DataManager, Vectorizer
from ml.model import Pytorch_model, BinaryTextClassifier
from config import MLConfig, FlaskConfig


app = Flask(__name__)
app.config.from_object(FlaskConfig)
data_manager = DataManager(max_lenght = MLConfig.MAX_SEQ_LENGHT)
model = Pytorch_model(MLConfig.MODEL_PATH, **MLConfig.MODEL_PARAMS)

from app import routes 