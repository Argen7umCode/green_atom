from bs4 import BeautifulSoup
from pprint import pprint
import requests
import random
from json import dump
from tqdm import tqdm
from nltk import flatten

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
]

imdb_link = 'https://www.imdb.com/'
headers = {'User-Agent': random.choice(user_agents)}

s = requests.Session()
s.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'})
start_page = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'
start_page_text = s.get(start_page).text


template = r'https://www.imdb.com'

soup = BeautifulSoup(start_page_text, 'html.parser')

def get_films_links():
    films_links = []
    for tag in tqdm(soup.find('ul', class_='ipc-metadata-list').findAll('a', class_='ipc-title-link-wrapper')):
        films_links.append('/'.join(tag.get('href').split('/')[:-1]))
    return films_links

def get_reviews_link(link):
    link = template + link + '/reviews?spoiler=hide&sort=curated&dir=desc&ratingFilter=0'
    return link

def get_reviews_links(links):
    return list(map(get_reviews_link, tqdm(links)))

def get_reviews(link):
    page = s.get(link).text
    reviews = BeautifulSoup(page, 'html.parser').findAll('div', class_='review-container')[:10]

    revs = []
    for review in tqdm(reviews):
        try: 
            score = 1 if int(review.find('div', class_='ipl-ratings-bar').find('span').find('span').text) > 5 else 0
        except:
            continue
        text = review.find('div', class_='text').text
        # data = {

        revs.append({
            'text' : text, 
            'score': score
        })
    return revs

links =  get_films_links()

rev_links = get_reviews_links(links)

s = requests.Session()
s.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'})

all_reviews = flatten([get_reviews(review) for review in tqdm(rev_links)])

with open('test/imdb_revs.json', 'w') as file:
    dump(all_reviews, file, indent=4)

pprint(all_reviews)