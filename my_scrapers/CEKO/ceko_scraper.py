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

class CEKO_cs_dep(scrapy.Spider):
    name = 'cs'
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=52']

    total_cs_data = {}

    def parse(self, response):
        data = response.css('div#tabs_desc_647_1 p span::text').getall()
        data = [clean_text(i) for i in data]
        about_cs = {
                "Computer Science Department of CEK at a Glance description": data[1],
                "data on the scope of Computer Engineering by CEK": data[3],
                "Data on the Admission to computer science by CEK": data[5]
            }
        hod_data = response.css('div#tabs_desc_647_2 p span::text, div#tabs_desc_647_2 p span strong::text').getall()
        hod_data = [clean_text(i) for i in hod_data]
        about_hod = {
            "Name of the CS department HOD": hod_data[0],
            "Designation of the CS department HOD": hod_data[1],
            "Phone number of the CS department HOD": hod_data[2],
        }
        print(hod_data)

        faculty_data = response.css('table#tablepress-1 tbody tr td::text').getall()
        faculty_data = [clean_text(i) for i in faculty_data]
        faculty_list = []
        for i in range(1,len(faculty_data),3):
            faculty_list.append(
                f"faculty name is {faculty_data[i]} and their designation is {faculty_data[i+1]}"
            )
        print(faculty_list)
        self.total_cs_data = {
            "Information about the CS (Computer science) department of College of Engineering Kottarakkara":{
                "About the CS department": about_cs,
                "Data on the HOD of CS department": about_hod,
                "Information of the faculty of CS department": faculty_list
            }
        }

    def closed(self, response):
        with open('college_json_data/ceko.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_cs_data)
        with open('college_json_data/ceko.json', 'w') as f:
            json.dump(data,f,indent=4)

