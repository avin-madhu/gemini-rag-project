import scrapy
import json
import scrapy.spiderloader
import re

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

# spider to get the details from the admission section
class CECH_main_info(scrapy.Spider):
    name = 'main info'
    start_urls = ['https://cecherthala.ihrd.ac.in/']

    total_info = {}
    def parse(self, response):
        about_college_desc = response.css('div#sppb-addon-1574412783958 div div.sppb-addon-content div::text').get()
        seat_info = [
                "BTech in Computer Science has 129 seats",
                "BTech in Electronics has 60 seats",
                "BTech in AI & DS has 60 seats",
                "BTech in Electrical Engineering has 30 seats",
                "MCA has 60 seats",
            ]
        contact_info ={
            "Address of the college" : "College of Engineering, Cherthala Pallippuram P.O, Alappuzha Dt.Kerala State, INDIA-688541",
            "Email of the principal": "principal@cectl.ac.in",
            "Email of the college" : "office@cectl.ac.in"
        }
        print(contact_info)
        self.total_info["About the college ( A basic description)"] = about_college_desc
        self.total_info["Information about the number of seats in various course or departments in the college"] = seat_info
        self.total_info["Link of the Brochure of the College"] = "https://cecherthala.ihrd.ac.in/images/2024/brochure_compressed.pdf"

    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)   

class CECH_transportation(scrapy.Spider):
    name = 'transportation'
    start_urls = ['https://cecherthala.ihrd.ac.in/index.php/about/facilities/transportation']

    total_transport_info = {}
    def parse(self, response):
        desc = response.css('div#column-id-1577005721711 div div#sppb-addon-1576926944817 div div div::text').getall()
        desc = [clean_text(i) for i in desc if i not in ['\n', '', ' ', '\xa0']]
        print(desc)
        trans_con = desc[3:]
        desc = desc[:3]
        self.total_transport_info["Transportation facilities detail in CEC ( Cherathala College )"] = desc
        self.total_transport_info["Transportation contact and faculty infoermation"] = trans_con
        
            
    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_transport_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)    

class CECH_hostel(scrapy.Spider):
    name = 'hostel'
    start_urls = ['https://cecherthala.ihrd.ac.in/index.php/about/facilities/hostel']

    total_hostel_info = {}
    def parse(self, response):
        desc = response.css('div#sppb-addon-1570898091086 div div div::text').getall()
        print(desc)
        self.total_hostel_info["Information about the hostel facility of College of Engineering Cherthala"] = desc[0] + desc[1]
        self.total_hostel_info["Information about hostel Contact and Fcaulty"] = {
            "For Girls Hostel": clean_text(desc[4]),
            "For Boys Hostel": clean_text(desc[5])
        }
    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_hostel_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)    


class CECH_principal(scrapy.Spider):
    name = 'principal'
    start_urls = ['https://cecherthala.ihrd.ac.in/index.php/administration/principal']

    total_principal_info = {}
    def parse(self, response):
        prin = {}
        data = response.css('li.TYR86d.wXCUfe.zfr3Q span::text, li.TYR86d.wXCUfe.zfr3Q span strong::text').getall()
        prin["Principal name"] = data[0]
        prin["Principal Designation"] = data[1]
        prin["Qualifications of the principal"] = [
            data[2]+data[3]+' '+data[4],
            data[5]+data[6]+' '+data[7],
            data[8]
        ]
        contact = response.css('table tbody tr td p a::text').getall()
        print(contact)
        prin["Contact Information of the Principal"] = {
            "phone number of prinicipal": contact[0],
            "email of principal": "principal@cectl.ac.in",
        }

        self.total_principal_info = prin

    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_principal_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)    

class CECH_undergraduate(scrapy.Spider):
    name = 'undergraduate'
    start_urls = ['https://cecherthala.ihrd.ac.in/index.php/courses/ug-programs']

    total_undergraduate_info = {}

    def parse(self, response):
        content = response.css('div.sppb-addon-content div::text').getall()
        content = [clean_text(i) for i in content]
        about_ug = ' '.join(content)
        uls = response.css('ul')
        programmes = []
        for ul in uls:
            programme = ul.css('li::text').getall()
            programme = [clean_text(i) for i in programme if i not in ['\n', '', ' ', '\xa0']]
            print(programme)
            if programme and programme[0] and len(programme)==1:
                programmes.append(programme[0])
        self.total_undergraduate_info["Information or decription about the UG programs or courses offered by CEC"] = about_ug
        self.total_undergraduate_info["These are the main UG(undergraduate) courses offered"] = programmes
        
            
    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_undergraduate_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)  

class CECH_postgraduate(scrapy.Spider):
    name = 'postgraduate'
    start_urls = ['https://cecherthala.ihrd.ac.in/index.php/courses/pg-programs']

    total_postgraduate_info = {}

    def parse(self, response):
        content = response.css('div.sppb-addon-content::text').getall()
        print(content)
        content = [clean_text(i) for i in content]
        about_pg = ' '.join(content)
        uls = response.css('ul')
        programmes = []
        for ul in uls:
            programme = ul.css('li::text').getall()
            programme = [clean_text(i) for i in programme if i not in ['\n', '', ' ', '\xa0']]
            print(programme)
            if programme and programme[0] and len(programme)==1:
                programmes.append(programme[0])
        self.total_postgraduate_info["Information or decription about the PG programs or courses offered by CEC"] = about_pg
        self.total_postgraduate_info["These are the main PG(postgraduate) courses offered"] = programmes
        
            
    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_postgraduate_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)  

class CECH_fees_structure(scrapy.Spider):
    name = 'fees_structure'
    start_urls = ['https://cecherthala.ihrd.ac.in/index.php/courses/admission/ug-admission']

    total_fees_info = {}

    def parse(self, response):
        content = response.css('div.sppb-addon-content::text,div.sppb-addon-content p::text').getall()
        content = [clean_text(i) for i in content if i not in ['\n']]
        print(content)
        admission = {}
        admission["Basic description about the admission at College of Engineering Cherthala"] = content[0] + ' '+content[1] + ' ' + content[2]
        admission["Information about the Academic Eligibilty at CEC"] = content[3] + ' ' + content[4] + ' ' + content[5]
        admission["Information about the fees structure at Colleg of Engineering Cherthala"] = {
            "Fees for Merit Regulated": content[6] + ' ' + content[7],
            "Fees for Merit Full": content[8] + ' ' + content[9],
            "Fees for NRI Seats": content[10] + ' ' + content[11]
        }
        self.total_fees_info["Information about te Admission and fees structure of College of Engineering Chengannur"] = admission
        
            
    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_fees_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)  

class CECH_DepartmentSpider(scrapy.Spider):
    name = 'departments'
    start_urls = ['http://www.cectl.ac.in/index.php/department/applied-engineering', 
                  'http://www.cectl.ac.in/index.php/department/computer-engineering', 
                  'http://www.cectl.ac.in/index.php/department/electronics-engineering',
                  'http://www.cectl.ac.in/index.php/department/electrical-engineering',
                  'http://www.cectl.ac.in/index.php/department/general-engineering']
    
    department_info = []

    def parse(self, response):
        # Extract department information
        department_name = response.css('h2.sppb-addon-title::text').get()
        print(department_name)
        print()
        department_desc = response.css('div#sppb-addon-1576923373664 div div div::text, div#sppb-addon-1576924354864 div div::text, div#sppb-addon-1576924221395 div div::text, div#sppb-addon-1576914754376 div div::text, div#sppb-addon-1576920054635 div div::text').getall()
        print(department_desc)
        hod_data = response.css('div#sppb-addon-1570521676233 div div span strong::text, div#sppb-addon-1570521676233 div div h3 span span::text, div#sppb-addon-1576914754348 div div h2::text, div#sppb-addon-1715674276172 div div h2::text').getall()
        hod_name = hod_data[0]
        print(hod_data)
        self.department_info.append({
            f"Information (description) about {department_name} department of CEC": department_desc,
            f"Name of the Head of the {department_name} department of CEC": hod_name
        })

    def closed(self, response):
        with open('college_json_data/cech.json', 'r') as f:
            try:
                data = json.load(f)
            except FileNotFoundError:
                data = []
            data.append(self.department_info)
        with open('college_json_data/cech.json', 'w') as f:
            json.dump(data,f,indent=4)