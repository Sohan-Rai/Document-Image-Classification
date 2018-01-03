# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 01:11:41 2017

@author: Sohan Rai
"""

from bs4 import BeautifulSoup
import requests
import re
import urllib
import os


def get_soup(url):
    return BeautifulSoup(requests.get(url).text)

image_type = "license"

query = "drivers license samples"
url = "http://www.bing.com/images/search?q=" + query + "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"

soup = get_soup(url)
images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]
for img in images:
    raw_img = urllib.request.urlopen(img).read()
    cntr = len([i for i in os.listdir("C:\\Users\\Sohan Rai\\Dropbox\\Studies\\Github codes\\images\\") if image_type in i]) + 1
    f = open("C:\\Users\\Sohan Rai\\Dropbox\\Studies\\Github codes\\images\\" + image_type + "_"+ str(cntr)+".jpeg", 'wb')
    f.write(raw_img)
    f.close()
