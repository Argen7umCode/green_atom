from app import app, data_manager, model
from flask import request, jsonify


@app.route('/')
def hello_geek():
    return '<h1>Hello from Flask & Docker</h2>'


@app.route('/predict', methods=['POST'])
def predict():
    # load text
    req = request.json
    text = req.get('text')
    
    # process text
    preprocessed_text = data_manager.data_preprocesser.preprocess_text(text)

    # vectorization 
    encoded_seq = data_manager.vectorizer.encode_sequence(preprocessed_text)
    padded_seq = data_manager.data_preprocesser.pad_seq(encoded_seq)

    # prediction
    predictions = model.predict(padded_seq)

    return jsonify({
        'predictions' : predictions,
        'result' : 1
    })


@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'code' : 'uspex'
    })