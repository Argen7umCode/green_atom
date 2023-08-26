import requests
from json import load

with open('test/imdb_revs.json') as file:
    reviews = load(file)


def out_red(text):
    print("\033[31m {}" .format(text))

def out_green(text):
    print("\033[32m {}" .format(text))

def out_blue(text):
    print("\033[34m {}" .format(text))

def get_test_pred_review(review):
    resp = requests.post('http://localhost:5000/predict', json={
        'text' : review
    }).json()
    return resp

correct = 0
for review in reviews:
    pred = get_test_pred_review(review.get('text')).get('predictions')
    score = review.get('score')
    if pred == score:
        out_green(f'pred: {pred}, score: {score}')
        correct += 1
    else:
        out_red(f'pred: {pred}, score: {score}')

out_blue(f'Total acc: {correct/len(reviews)}')