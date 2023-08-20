from flask import Flask, request, jsonify
from dataprocess import DataPreProcesser, Vectorizer

MAX_SEQ_LENGHT = 1000

app = Flask(__name__)
data_preprocesser = DataPreProcesser(MAX_SEQ_LENGHT)
vectorizer = Vectorizer()


@app.route('/predict', methods=['POST'])
def predict():
    # load text
    
    req = request.json
    text = req.get('text')
    
    
    # process text
    preprocessed_text = data_preprocesser.preprocess_text(text)

    # vectorization 
    encoded_seq = vectorizer.encode_sequence(preprocessed_text)
    padded_seq = data_preprocesser.pad_seq(encoded_seq)
    # prediction

    # return json


    return jsonify({
        'preprocessed_text' : preprocessed_text,
        'result' : 1
    })

