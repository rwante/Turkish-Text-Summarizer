import requests
from bs4 import BeautifulSoup


def webscraper(url):
    htmldata = requests.get(url).text
    soup = BeautifulSoup(htmldata, 'lxml')
    data_ls = []
    for data in soup.find_all("p"):
        data_ls.append(data.get_text())
    return " ".join(data_ls)


