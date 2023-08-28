import requests


res = requests.get(url='http://localhost:5000')

print(res.text)