from flask import Flask, request, jsonify
from ml.dataprocess import DataPreProcesser, Vectorizer, DataImporter
from config import MAX_SEQ_LENGHT, Config, basedir


app = Flask(__name__)
app.config.from_object(Config)
data_preprocesser = DataPreProcesser(MAX_SEQ_LENGHT)
data_importer = DataImporter()
vectorizer = Vectorizer()


from app import routes