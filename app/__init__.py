from flask import Flask
from ml.dataprocess import DataPreProcesser, Vectorizer
from ml.model import Pytorch_model
from config import MLConfig, FlaskConfig


app = Flask(__name__)
app.config.from_object(FlaskConfig)
data_importer = DataImporter()
data_preprocesser = DataPreProcesser(MLConfig.MAX_SEQ_LENGHT)

vectorizer = Vectorizer()
model = Pytorch_model(MLConfig.MODEL_PATH)

from app import routes 