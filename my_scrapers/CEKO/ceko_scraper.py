import scrapy
import json
import re
import scrapy.crawler

def clean_text(item):
    item = re.sub(r'\\u[0-9a-fA-F]{4}', '', item)
    item = re.sub(r'[\u2013\u2019\u2022\u201a\u201c\u201d\u00b1\xa0]', '-', item)
    item = re.sub(r'\s+', ' ', item).strip()
    item = re.sub(r'<br>','',item)
    item = re.sub(r'-','',item).strip()
    return item

class CEKO_principal(scrapy.Spider):
    name = 'principal'
    # allowed_domains = ['cea.ac.in']
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=1717']

    total_principal_data = {}

    def parse(self, response):
        data = response.css('div.entry-content p::text, div.entry-content p span strong::text, div.entry-content p strong::text').getall()
        data = [clean_text(i) for i in data]
        print(data)
        principal_data = {
            'principal_name': data[0] + data[1],
            'Principal of college name' : data[2],
            'Contact Number of principal' : data[3],
            'Email of principal': "cekottarakkara.ihrd@gmail.com"
        }
        self.total_principal_data["Data about the principal of College of Engineering Kottarakkara"] = principal_data
        

    def closed(self, response):
        with open('college_json_data/ceko.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_principal_data)
        with open('college_json_data/ceko.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKO_about_the_college(scrapy.Spider):
    name = 'about'
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=64']

    total_about_data = {}

    def parse(self, response):
        data = response.css('div.entry-content p span::text, div.entry-content p a::text').getall()
        data = [clean_text(i) for i in data if i not in ['.\xa0']]
        self.total_about_data["General Data about College of Engineering Kottarakkara"] = ' '.join(data)
        print(data)

    def closed(self, response):
        with open('college_json_data/ceko.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_about_data)
        with open('college_json_data/ceko.json', 'w') as f:
            json.dump(data,f,indent=4)