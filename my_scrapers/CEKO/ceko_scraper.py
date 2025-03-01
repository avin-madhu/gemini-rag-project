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
        # print(faculty_list)


        data = response.css('div#tabs_desc_647_4 li span::text').getall()
        print(data)

        self.total_cs_data = {
            "Information about the CS (Computer science) department of College of Engineering Kottarakkara":{
                "About the CS department": about_cs,
                "Data on the HOD of CS department": about_hod,
                "Information of the faculty of CS department": faculty_list,
                "Information about the UG (undergraduate) course offered by College Engineering Kottarakkara": {
                    "The Department offers these courses": data[:3],
                    "seat information for UG Course": ' '.join(data[3:])
                }
            }
        }

    def closed(self, response):
        with open('college_json_data/ceko.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_cs_data)
        with open('college_json_data/ceko.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKO_ec_dep(scrapy.Spider):
    name = 'ec'
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=50']  # Replace with the actual URL of the EC department page

    total_ec_data = {}

    def parse(self, response):
        data = response.css('div#tabs_desc_664_1 p span::text').getall()
        data = [clean_text(i) for i in data]
        # print(data)
        about_ec = {
                "Electronics and Communication Department of CEK at a Glance description": data[0],
                "data on the admission to Electronics and Communication Engineering by CEK": data[1],
            }
        hod_data = response.css('div#tabs_desc_664_2 p span::text, div#tabs_desc_664_2 p span strong::text').getall()
        hod_data = [clean_text(i) for i in hod_data]
        print(hod_data, "HOD DATA")
        about_hod = {
            "Name of the EC department HOD": hod_data[0],
            "Designation of the EC department HOD": hod_data[1],
            "Phone number of the EC department HOD": hod_data[2],
        }

        faculty_data = response.css('table#tablepress-2 tbody tr td::text').getall()
        faculty_data = [clean_text(i) for i in faculty_data]
        faculty_list = []
        for i in range(1,len(faculty_data),3):
            faculty_list.append(
                f"faculty name is {faculty_data[i]} and their designation is {faculty_data[i+1]}"
            )
        print(faculty_list)
        self.total_ec_data = {
            "Information about the EC (Electronics and Communication) department of College of Engineering Kottarakkara":{
                "About the EC department": about_ec,
                "Data on the HOD of EC department": about_hod,
                "Information of the faculty of EC department": faculty_list
            }
        }

    def closed(self, response):
        with open('college_json_data/ceko.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_ec_data)
        with open('college_json_data/ceko.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKO_as_dep(scrapy.Spider):
    name = 'as'
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=54']  # Replace with the actual URL of the Applied Science department page

    total_as_data = {}

    def extract_data(self, response, css_selector):
        try:
            data = response.css(css_selector).getall()
            data = [clean_text(i) for i in data]
            return data
        except Exception as e:
            print(f"Error extracting data: {e}")
            return []

    def parse(self, response):
        hod_data = self.extract_data(response, 'div#tabs_desc_676_1 p::text, div#tabs_desc_676_1 p span::text, div#tabs_desc_676_1 p span strong::text')
        # print(hod_data)
        about_hod = {
            "Name of the Applied Science department HOD": hod_data[0],
            "Designation of the Applied Science department HOD": hod_data[1],
            "Phone number of the Applied Science department HOD": hod_data[2],
        }

        faculty_data = self.extract_data(response, 'table#tablepress-3 tbody tr td::text')
        faculty_list = []
        for i in range(1, len(faculty_data), 3):
            faculty_list.append(
                f"faculty name is {faculty_data[i]} and their designation is {faculty_data[i+1]}"
            )
        print(faculty_data, "FD")
        self.total_as_data = {
            "Information about the Applied Science department of College of Engineering Kottarakkara": {
                "Data on the HOD of Applied Science department": about_hod,
                "Information of the faculty of Applied Science department": faculty_list
            }
        }

    def closed(self, response):
        try:
            with open('college_json_data/ceko.json', 'r') as f:
                data = json.load(f)
                data.append(self.total_as_data)
            with open('college_json_data/ceko.json', 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error storing data: {e}")

class CEKO_ge_dep(scrapy.Spider):
    name = 'general_engineering'
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=1669']  # Replace with the actual URL of the General Engineering department page

    total_general_engineering_data = {}

    def extract_data(self, response, css_selector):
        try:
            data = response.css(css_selector).getall()
            data = [clean_text(i) for i in data]
            return data
        except Exception as e:
            print(f"Error extracting data: {e}")
            return []

    def parse(self, response):
        hod_data = self.extract_data(response, 'div#tab-content_1672 p::text, div#tab-content_1672 p span::text, div#tab-content_1672 p span strong::text')
        print(hod_data, "HOD DATA")
        about_hod = {
            "Name of the General Engineering department HOD": hod_data[0],
            "Designation of the General Engineering department HOD": hod_data[1],
            "Phone number of the General Engineering department HOD": hod_data[2],
        }

        faculty_data = self.extract_data(response, 'div#tabs_desc_1672_2 table tbody tr td::text')
        faculty_list = []
        for i in range(4, len(faculty_data), 3):
            faculty_list.append(
                f"faculty name is {faculty_data[i]} and their designation is {faculty_data[i+1]}"
            )

        self.total_general_engineering_data = {
            "Information about the General Engineering department of College of Engineering Kottarakkara": {
                "Data on the HOD of General Engineering department": about_hod,
                "Information of the faculty of General Engineering department": faculty_list
            }
        }

    def closed(self, response):
        try:
            with open('college_json_data/ceko.json', 'r') as f:
                data = json.load(f)
                data.append(self.total_general_engineering_data)
            with open('college_json_data/ceko.json', 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error storing data: {e}")


class CEKO_admission(scrapy.Spider):
    name = 'admission'
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=123']  # Replace with the actual URL of the Admission page

    total_admission_data = {}

    def parse(self, response):
        link_1 = " https://cekottarakkara.ihrd.ac.in/wp-content/uploads/2024/08/List-of-Documents-2024-admission.pdf"
        link_2 = "https://cekottarakkara.ihrd.ac.in/wp-content/uploads/2024/08/Admission-fee-2024.pdf"
        link_3 = "https://forms.gle/bBUwKZPLSPm58XgT7"
        link_4 = "https://nri.ihrd.ac.in/"
        link_5 = "https://cekottarakkara.ihrd.ac.in/wp-content/uploads/2021/11/Vrequest-TC-R-converted.pdf"
        link_6 = "https://cekottarakkara.ihrd.ac.in/wp-content/uploads/2024/07/LET-admission-2024-allotment-fee-to-be-paid.pdf"
        link_7 = "https://cekottarakkara.ihrd.ac.in/wp-content/uploads/2023/08/List-of-Documents.pdf"

        self.total_admission_data = {
            "Link for the documents required for BTECH admission at CEK": link_1,
            "Details for the FEES to be paid for BTECH admission in CEK": link_2, 
            "Link for the registration of NON-KEAM 2024 admission for BTECH at CEK": link_3, 
            "Link for admission of NRI Students at CEK": link_4, 
            "Link for the request of TC ( Termiination Certficate )": link_5,
            "Fee details for LET (Lateral Entry Students) at college of Engineering Kottarakkara": link_6,
            "List of documents to be submitted for LET (Lateral Entry Students) Admission at CEK": link_7
        }
    def closed(self, response):
        try:
            with open('college_json_data/ceko.json', 'r') as f:
                data = json.load(f)
                data.append(self.total_admission_data)
            with open('college_json_data/ceko.json', 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error storing data: {e}")


class CEKO_placement(scrapy.Spider):
    name = 'placement'
    start_urls = ['https://cekottarakkara.ihrd.ac.in/?page_id=1057']  # Replace with the actual URL of the Admission page

    total_placement_data = {}

    def parse(self, response):
        data = response.css('div.elementor-text-editor.elementor-clearfix p strong::text').getall()
        data = [clean_text(i) for i in data]
        print(data)
        self.total_placement_data = {
            "Placement officer of College of Engineering Kottarakkara": data[1]
        }

    def closed(self, response):
        try:
            with open('college_json_data/ceko.json', 'r') as f:
                data = json.load(f)
                data.append(self.total_placement_data)
            with open('college_json_data/ceko.json', 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error storing data: {e}")










