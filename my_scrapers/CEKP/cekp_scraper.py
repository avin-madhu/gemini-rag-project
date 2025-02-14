import scrapy
import json
import re

import scrapy.crawler

def clean_text(item):
    item = re.sub(r'\\u[0-9a-fA-F]{4}', '', item)
    item = re.sub(r'[\u2013\u2019\u2022\u201a\u201c\u201d\u00b1\u00a0]', '-', item)
    item = re.sub(r'\s+', ' ', item).strip()
    item = re.sub(r'<br>','',item)
    item = re.sub(r'<strong>','',item)
    item = re.sub(r'\u00a0','',item)
    item = re.sub(r'-','',item).strip()
    item = re.sub(r'[^\x00-\x7F]+', '', item)
    return item

class CEKP_principal(scrapy.Spider):
    name = 'principal'
    start_urls = ['https://www.cek.ac.in/index.php/administration/principal']

    total_principal_data = {}

    def parse(self, response):
        principal_data = response.css('p.MsoNormal span::text, p.MsoNormal span::text').getall()
        cleaned_principal_data = [clean_text(i) for i in principal_data]
        data = {}
        data['Principal Name'] = cleaned_principal_data[0]
        data['principal Desgination'] = cleaned_principal_data[1]
        data["phone or contact number of the principal"] = cleaned_principal_data[2]
        data['fax Number of the principal'] = cleaned_principal_data[3]
        data['Mobile Number of the principal'] = cleaned_principal_data[5]
        data['Email ID of the principal'] = "principal@cek.ac.in"
        self.total_principal_data['information about the principal'] = data

    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_principal_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKP_admission_mtech(scrapy.Spider):
    name = 'admission'
    start_urls = ['https://www.cek.ac.in/index.php/academics/admission/m-tech']

    total_admission_data = {
        "Basic data on the courses or programmes offered": "College of Engineering Kalloopaara offers two programmes or courses for students: Btech and Mtech"
    }
    

    def parse(self, response):
        # Extract course information
        course_text = response.css('title::text').get()
        
        data = response.css('p.MsoNormal b span::text').getall()
        desc = response.css('p.MsoNormal span::text').getall()
        clean_data = [clean_text(i) for i in data]
        clean_desc = [clean_text(i) for i in desc]

        desc_new = ' '.join(clean_desc)

        course_name = clean_data[1]
        course_seats = clean_data[2] + 'seats'
        desciption = desc_new
        
        print(clean_data)

        # Extract important links
        dte_link = response.xpath('//a[contains(@href, "dtekerala.gov.in")]/@href').get()
        portal_link = response.xpath('//a[contains(@href, "dtekerala.co.in/site/login")]/@href').get()

        # Extract description paragraphs
        description = []
        for para in response.xpath('//p[@class="MsoNormal" and contains(., "eligible for admission")]/text()').getall():
            description.append(para.strip())

        self.total_admission_data['Information about the Mtech Programme in Kalloopaara'] = {
            "Course Name offered in Mtech": course_name,
            'Number of seats in Mtech course': course_seats,
            'Description of the Mtech course': desciption,
            'dte_website': dte_link,
            'admission_portal for Mtech Course': portal_link,
            'description about Mtech Course or Programme': ' '.join(description),
            'university under': "APJ Abdul Kalam Technological University (KTU)",
            'admission_process of Mtech Course or progammes': "Through Directorate of Technical Education Kerala",
        }

    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_admission_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)


class CEKP_admission_btech(scrapy.Spider):
    name = 'btech_admission'
    start_urls = ['https://www.cek.ac.in/index.php/academics/admission/b-tech']
    
    total_btech_admission_data = {}

    def parse(self, response):
        # Extract the course information
        courses = response.css('ul li span::text').getall()
        courses = [course.strip() for course in courses]
        courses = [clean_text(courses[i]) for i in range(6,len(courses))]

        print(courses)

        other_data = response.css('p.MsoNormal span::text').getall()
        other_data = [clean_text(other_data[i]) for i in range(0, len(other_data))]
        other_data = [i for i in other_data if i]

        total_seat_intake = other_data[4]

        # Extract the seat information
        seat_info = [other_data[i] for i in range(5,9)]
        seat_info = ' and '.join(seat_info)
        print(seat_info, "seat")

        # Extract the eligibility information
        eligibility_info = other_data[16] + other_data[19]

        # Yield the extracted information
        self.total_btech_admission_data = {
            'Courses or Programmes Offered in Btech by kalloopaara ': courses,
            'Total Seat intake in Btech Information': total_seat_intake,
            'Seat in Btech information': seat_info,
            'Btech Critiriea for Eligibility and also Age critiriea': eligibility_info,
        }
    
    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_btech_admission_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKP_department_ec(scrapy.Spider):
    name = "department_ec"
    start_urls = [
        'https://cek.ac.in/index.php/departments/electronics-and-communication-engineering',  # Replace with the actual URL
    ]

    total_ec_data = {}

    def parse(self, response):
        # Extract the department head information
        hod_and_faculty_data = response.css('p.MsoNormal span::text, p.MsoNormal b span::text').getall()
        desc = response.css('p span::text').getall()
        print(desc)
        hod_and_faculty_data = [clean_text(i) for i in hod_and_faculty_data]
        print(hod_and_faculty_data)
        hod_data = {
            "Name of Hod of EC department": hod_and_faculty_data[0],
            "Designation of HOD of EC Department": hod_and_faculty_data[1],
            "Phone Number and Email of Hod of EC Department": hod_and_faculty_data[2] + "and Email is philipcherian@cek.ac.in and hodece@cek.ac.in"
        }
        print(hod_and_faculty_data[12])
        count = 1
        faculty_details = []
        for i in range(12, len(hod_and_faculty_data),3):
            faculty_details.append(f"Name: {hod_and_faculty_data[i+1]} and his/her designation is {hod_and_faculty_data[i+2]}")
            count += 1
        self.total_ec_data["Data about the Electronics Department"] = {
            "Data about the HOD of EC": hod_data,
            "Data about the Faculty of EC": faculty_details,
            "Description of EC Department": desc[-6]
        }

    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_ec_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKP_department_cs(scrapy.Spider):
    name = "department_cs"
    start_urls = [
        'https://cek.ac.in/index.php/departments/computer-science-and-engineering',  # Replace with the actual URL
    ]

    total_cs_data = {}

    def parse(self, response):
        # Extract the department head information
        data = response.css('p.MsoNormal span::text, p.MsoNormal b span::text').getall()
        desc = ' '.join([clean_text(data[i]) for i in range(0, 5)])
        print(desc)
        data = [clean_text(i) for i in data]
        print(data)
        hod_data = {
            "Name of Hod of CS department": data[6],
            "Designation of HOD of CS Department": data[7],
            "Phone Number and Email of Hod of CS Department": data[8] + "and Email is hodcse@cek.ac.in"
        }
        faculty_details = response.css('table.MsoNormalTable tbody tr td p.MsoNormal span::text').getall()
        faculty_details = [clean_text(i) for i in faculty_details]
        faculty_details = faculty_details[11:]
        faculty_details = [i for i in faculty_details if i not in ['', 'Mrs.']]
        print(faculty_details)
        fac_data = []
        for i in range(0,len(faculty_details),3):
            fac_data.append(
                   f"Faculty name is {faculty_details[i+1]} and his/her designation is {faculty_details[i+2]}"
            )
        self.total_cs_data["Information (Names and designation) or details about the faculty of CS (computer Science) Department"] = fac_data
        self.total_cs_data["Information about the HOD ( Head of department) of Computer Science (CS)"] = hod_data
        self.total_cs_data["Description about the CS Department"] = desc

    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_cs_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKP_department_cs_cyber(scrapy.Spider):
    name = "department_cs_cyber"
    start_urls = [
        'https://cek.ac.in/index.php/departments/computer-science-and-engineering-2',  # Replace with the actual URL
    ]

    total_cs_data = {}

    def parse(self, response):
        # Extract the department head information
        data = response.css('p.MsoNormal span::text, p.MsoNormal b span::text').getall()
        desc = clean_text(data[2])
        no_of_seats = data[4]
        data = [clean_text(i) for i in data]
        print(data)
        hod_data = {
            "Name of Hod of CyberSecurity department": data[7],
            "Designation of HOD of CyberSecurity Department": data[8],
            "Phone Number and Email of Hod of CyberSecurity Department": data[9] + "and Email is hodcc@cek.ac.in"
        }
        fac_data = []
        for i in range(20,54,7):
            fac_data.append(
                   f"Faculty name is {data[i+1]} and his/her designation is {data[i+2]}"
            )
        self.total_cs_data["Information (Names and designation) or details about the faculty of CS - CyberSecurity Department"] = fac_data
        self.total_cs_data["Information about the HOD ( Head of department) of Computer Science-(CyberSecurity)"] = hod_data
        self.total_cs_data["Description about the Cyber Security Department"] = desc
        self.total_cs_data["Number of seats of Cyber Security Department"] = no_of_seats

    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_cs_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKP_department_eee(scrapy.Spider):
    name = "department_eee"
    start_urls = [
        'https://cek.ac.in/index.php/departments/electrical-and-electronics-engineering',
    ]

    total_eee_data = {}

    def parse(self, response):
        # Extract the department head information
        data = response.css('p.MsoNormal span::text, p.MsoNormal b span::text').getall()
        desc = clean_text(data[2])
        data = [clean_text(i) for i in data]
        data = [i for i in data if i]
        print(data)
        hod_data = {
            "Name of Hod of Electrical and Electronics Engineering department": data[4],
            "Designation of HOD of Electrical and Electronics Engineering Department": data[5],
            "Phone Number of Hod of Electrical and Electronics Engineering Department": data[6]
        }
        fac_data = []
        for i in range(12,24,3):
            fac_data.append(
                   f"Faculty name is {data[i+1]} and his/her designation is {data[i+2]}"
            )
        self.total_eee_data["Information (Names and designation) or details about the faculty of Electrical and Electronics Engineering Department"] = fac_data
        self.total_eee_data["Information about the HOD ( Head of department) of Electrical and Electronics Engineering"] = hod_data
        self.total_eee_data["Description about the Electrical and Electronics Engineering Department"] = desc
        # self.total_eee_data = data
    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_eee_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEKP_department_geas(scrapy.Spider):
    name = "department_geas"
    start_urls = [
        'https://cek.ac.in/index.php/departments/general-engineering-applied-sciences',
    ]

    total_geas_data = {}

    def parse(self, response):
        # Extract the department head information
        data = response.css('p.MsoNormal span::text, p.MsoNormal b span::text').getall()
        desc = clean_text(data[1])
        data = [clean_text(i) for i in data]
        data = [i for i in data if i]
        print(data)
        hod_data = {
            "Name of Hod of General Engineering and Applied Sciences department": data[3],
            "Designation of HOD of General Engineering and Applied Sciences Department": data[4],
            "Phone Number and Email of Hod of General Engineering and Applied Sciences Department": data[5]
        }
        fac_data = []
        for i in range(10,21,3):
            fac_data.append(
                   f"Faculty name is {data[i+1]} and his/her designation is {data[i+2]}"
            )
        self.total_geas_data["Information (Names and designation) or details about the faculty of General Engineering and Applied Sciences Department"] = fac_data
        self.total_geas_data["Information about the HOD ( Head of department) of General Engineering and Applied Sciences"] = hod_data
        self.total_geas_data["Description about the General Engineering and Applied Sciences Department"] = desc
    def closed(self, response):
        with open('college_json_data/cekp.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_geas_data)
        with open('college_json_data/cekp.json', 'w') as f:
            json.dump(data,f,indent=4)
