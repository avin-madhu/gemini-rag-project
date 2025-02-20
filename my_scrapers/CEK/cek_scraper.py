# import scrapy
# import json
# import re

# import scrapy.crawler

# def clean_text(item):
#     item = re.sub(r'\\u[0-9a-fA-F]{4}', '', item)
#     item = re.sub(r'[\u2013\u2019\u2022\u201a\u201c\u201d\u00b1\xa0]', '-', item)
#     item = re.sub(r'\s+', ' ', item).strip()
#     item = re.sub(r'<br>','',item)
#     item = re.sub(r'-','',item).strip()
#     return item

# class CEK_principal(scrapy.Spider):
#     name = 'principal'
#     # allowed_domains = ['cea.ac.in']
#     start_urls = ['https://ceknpy.ac.in/principal']

#     total_principal_data = {}

#     def parse(self, response):
#         principal_data = {}
#         principal_name = response.css('table tbody tr td p span strong::text').get()
#         principal_other_data = response.css('table tbody tr td p span::text').getall()
#         print(principal_name)
#         print(principal_other_data[:5])
#         principal_data["Principal Name"] = principal_name
#         principal_data["Department of the principal"] = principal_other_data[0]
#         principal_data["Address of the principal"] = {"college":principal_other_data[1],"place and PIN Code": principal_other_data[2]} 
#         principal_data["Phone Number and Fax Number of the Principal"] = principal_other_data[3]
#         principal_data["Email ID of the principal"] = principal_other_data[4]
#         self.total_principal_data["Data about the Principal of CEK"] = principal_data

#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_principal_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_management(scrapy.Spider):
#     name = 'board_members'
#     # allowed_domains = ['cea.ac.in']
#     start_urls = ['https://www.ceknpy.ac.in/management']

#     total_management_data = {}

#     def parse(self, response):
#         member_data = response.css('div.sec-title div p span::text').getall()
#         print(member_data)
#         management_data = ' '.join(member_data)
#         self.total_management_data["Data about Management of CEK"] = management_data

#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_management_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_admin_staff(scrapy.Spider):
#     name = 'admin_staff'
#     # allowed_domains = ['cea.ac.in']
#     start_urls = ['https://www.ceknpy.ac.in/adminstaff']

#     total_admin_staff = {}

#     def parse(self, response):
#         final_admin_staff_data = []
#         admin_staff_data = response.css('div.sec-title div table tr p::text').getall()
#         cleaned_admin_staff_data = []
#         for i in admin_staff_data:
#             if i not in ['\xa0','\r\n']:
#                 cleaned_admin_staff_data.append(i)

#         print(cleaned_admin_staff_data)
#         i = 0
#         while(i<len(cleaned_admin_staff_data)):
#             data = {}
#             data["Name"] = clean_text(cleaned_admin_staff_data[i])
#             data["Designation"] = clean_text(cleaned_admin_staff_data[i+1])
#             data["E-mail"] = clean_text(cleaned_admin_staff_data[i+2])
#             data["Phone Number"] = clean_text(cleaned_admin_staff_data[i+3])
#             i+=4
#             final_admin_staff_data.append(data)
#         self.total_admin_staff["Admnistration staff Data"] = final_admin_staff_data
        
#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_admin_staff)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_overview(scrapy.Spider):
#     name = 'overview'
#     # allowed_domains = ['cea.ac.in']
#     start_urls = ['https://www.ceknpy.ac.in/overview']

#     total_overview_data = ''

#     def parse(self, response):
#         data = {}
#         overview_data = response.css('div.sec-title div.desc p::text, div.sec-title div::text').getall()
#         for i in overview_data:
#             i.replace('\r\n','')
#         data["Overview of the college (CEK)"] = clean_text(overview_data[1] + ' ' + overview_data[2])
#         data["Vision of CEK"] = clean_text(overview_data[3])
#         data["Mission"] = clean_text(overview_data[4])

#         data["Programmes Offered"] = {
#             "Btech": [clean_text(overview_data[5]), clean_text(overview_data[6]), clean_text(overview_data[7]), clean_text(overview_data[8])],
#             "Mtech": clean_text(overview_data[10])
#         }
#         data["Memorandum of Understanding"] = clean_text(overview_data[11].replace('\r',''))
#         data["Student Professional Bodies"] = clean_text(overview_data[12])
#         data["Data about College Library"] = clean_text(overview_data[13])
#         data["Training and Placement Cell of CEK"] = clean_text(overview_data[14])
#         data["Alumni details about CEK"] = clean_text(overview_data[15])
#         data["Parent-Teacher Association (PTA) of CEK"] = clean_text(overview_data[16])
#         data["College Hostel"] = clean_text(overview_data[17])
#         data["Canteen"] = clean_text(overview_data[18])
#         data["ATM Counter"] =  clean_text(overview_data[19])
#         data["Sports and Cultural Festivals"] = clean_text(overview_data[20])
#         self.total_overview_data = {
#             "About the Overview of the college":data
#         }

#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_overview_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_infrastructure(scrapy.Spider):
#     name = 'infrastructure'
#     start_urls = ['https://www.ceknpy.ac.in/infrastructure']

#     total_infrastructure_data = {}

#     def parse(self, response):
#         data = {}
#         infra_data = response.css('div.sec-title div.desc p::text').getall()
#         for i in infra_data:
#             i.replace('\r\n','')

#         data["Data about building and Labs"] = clean_text(infra_data[0])
#         data["Information about the roads to the college (CEK)"] = infra_data[1]
#         data["About the placement activities"] = infra_data[2]
#         data["PTA (parent teacher association) at CEK"] = infra_data[3]
#         data["Transportation details at CEK"] = infra_data[4]
#         data["Water supply at CEK"] = infra_data[5]
#         self.total_infrastructure_data["About the Infrastructure at CEK"] = data

#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_infrastructure_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)
        
# class CEK_anti_ragging_squad(scrapy.Spider):
#     name = 'Anti_Ragging_squad'
#     # allowed_domains = ['cea.ac.in']
#     start_urls = ['https://www.ceknpy.ac.in/antiragging']

#     total_Anti_Ragging_Squad = {}

#     def parse(self, response):
#         data = []
#         ars_data = response.css('table tr td p span::text').getall()
#         print(ars_data[13:])
#         squad_list = ars_data[13:]
#         print(squad_list)
#         i = 0
#         while i<len(squad_list):
#             print(ars_data[i])
#             data.append(
#                 {
#                     "Name": squad_list[i],
#                     "Phone Number": squad_list[i+1]
#                 }
#             )
#             i+=2
#         self.total_Anti_Ragging_Squad["Data about the Anti Ragging Squad"] ={
#             "List of Executive Commitee Members in Anti Ragging Squad": data
#         } 
    
#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_Anti_Ragging_Squad)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_Departmentdata(scrapy.Spider):
#     name = 'department'
#     start_urls = ['https://www.ceknpy.ac.in/departments/'] 

#     total_department_data = {
#         "About the departments of CEK":[]
#     }  # Change to a list to hold multiple department data

#     def parse(self, response):
#         department_links = response.css('div.content-part span a::attr(href)').getall()
#         print(department_links)
#         for department_link in department_links:
#             yield response.follow(department_link, self.parse_department)  

#     def parse_department(self, response):
#         data = {}
#         course_overview_data = response.css('div.course-overview div p::text, div.course-overview div h3::text').getall()
#         department_title = response.css('div.breadcrumbs-text ul li::text').getall()
#         department_title = department_title[-1]
        
#         course_overview_data_cleaned = [clean_text(i) for i in course_overview_data if clean_text(i)]

#         # for CS department 
#         if(department_title == 'Department of Computer Science and Engineering'):
#             cs_dep_data = {}
#             temp = [course_overview_data_cleaned[i] for i in range(5)]
#             cs_dep_data["About the Computer Science Department"] = ' '.join(temp)

#             seat_table_data = {
#                 "BTech Computer Science": "120 Seats",
#                 "BTech Artificial Intelligence and Data Science": "60 Seats"
#             }
#             cs_dep_data["Information about the seats in CS departments"] = seat_table_data

#             programme_outcomes = [course_overview_data_cleaned[i] for i in range(5,35)]
#             cs_dep_data["Programme Outcome information of CS department"] = programme_outcomes
#             hod_data = {}
#             hod_data["Name"] = response.css("div.names::text").get()

#             # Getting the data of CS HOD
#             hod_qualifications = response.css('div#prod-curriculum div.right_con::text').getall()
#             hod_qualifications = [clean_text(i) for i in hod_qualifications]
#             hod_data["Qualifications of HOD"] = {
#                 "Qualificaion": hod_qualifications[0],
#                 "Department": hod_qualifications[1],
#                 "Years of Experience": hod_qualifications[2]
#             }
#             hod_data["Contact Information of HOD"] = {
#                 "Phone Number": hod_qualifications[3],
#                 "Email": hod_qualifications[4]
#             }
#             hod_publications = response.css('div#prod-curriculum div.inner-box div.box p::text').getall()
#             hod_publications = [clean_text(i) for i in hod_publications[13:]]
#             hod_data["Publications of HOD"] = hod_publications
#             cs_dep_data["Information on Head of Department of Computer Science"] = hod_data

#             # Getting the faculty list of CEK Computer Science
#             faculty_cards = response.css('div.card.accordion.block')
#             faculty_info = []
#             for card in faculty_cards:
#                 name = card.css("div.names::text").get()
#                 phone_number = card.css("div.right_con::text").re_first(r"\d{10}")
#                 email = card.css("div.right_con::text").re_first(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%+-]+\.[a-zA-Z]{2,}")
#                 designation = card.css("div.col-lg-8 div.designation::text").get()
#                 if name:
#                     faculty_info.append(
#                         {
#                             "Name": name,
#                             "Designation": clean_text(designation),
#                             "Phone Number": phone_number,
#                             "Email": email
#                         }
#                     )
#             cs_dep_data["Computer Science Faculty Information"] = faculty_info
#             course_overview_data_cleaned ={
#                 "Department of Applied Science": cs_dep_data
#             }
#             self.total_department_data["About the departments of CEK"].append(course_overview_data_cleaned)
        
#         # Getting the data of Electronice Department 
#         if(department_title == "Department of Electronics and Communication"):
#             ec_dep_data = {}
#             ec_dep_data["About the Electronics Department of CEK"] = ''.join([course_overview_data_cleaned[i] for i in range(5)])

#             hod_data = {}
#             hod_qualifications = response.css('div#prod-curriculum div.right_con::text').getall()
#             hod_qualifications = [clean_text(i) for i in hod_qualifications]
#             print(hod_qualifications)
#             if hod_qualifications[0]:
#                 hod_data["Qualifications of HOD"] = {
#                     "Qualificaion": hod_qualifications[0],
#                     "Department": hod_qualifications[1],
#                     "Years of Experience": hod_qualifications[2]
#                 }
#                 hod_data["Contact Information of HOD"] = {
#                     "Phone Number": hod_qualifications[3],
#                     "Email": hod_qualifications[4]
#                 }
#                 hod_publications = response.css('div#prod-curriculum div.inner-box div.box p::text').getall()
#                 hod_publications = [clean_text(i) for i in hod_publications[13:]]
#                 hod_data["Publications of HOD"] = hod_publications
#                 ec_dep_data["Information on Head of Department of Electronics Department"] = hod_data
#             else:
#                 hod_data = "None Specified"
#                 ec_dep_data["Information on Head of Department of Electronics Department"] = hod_data

#             # Getting the faculty information of Electronics department of CEK
#             faculty_cards = response.css('div.card.accordion.block')
#             faculty_info = []
#             for card in faculty_cards:
#                 name = card.css("div.names::text").get()
#                 phone_number = card.css("div.right_con::text").re_first(r"\d{10}")
#                 email = card.css("div.right_con::text").re_first(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%+-]+\.[a-zA-Z]{2,}")
#                 designation = card.css("div.col-lg-8 div.designation::text").get()
#                 if name:
#                     faculty_info.append(
#                         {
#                             "Name": name,
#                             "Designation": clean_text(designation),
#                             "Phone Number": phone_number,
#                             "Email": email
#                         }
#                     )
#             ec_dep_data["Electronics Faculty Information"] = faculty_info
#             course_overview_data_cleaned ={
#                 "Department of Applied Science": ec_dep_data
#             }
#             self.total_department_data["About the departments of CEK"].append(course_overview_data_cleaned)

#         if(department_title == "Department of Mechanical Engineering"):
#             mech_dep_data = {}
#             mech_dep_data["About the Mechanical Engineering Department of CEK"] = ''.join([course_overview_data_cleaned[i] for i in range(4)])

#             # Getting the information of HOD of mechanical Engineering
#             hod_data = {}
#             hod_qualifications = response.css('div#prod-curriculum div.right_con::text').getall()
#             hod_qualifications = [clean_text(i) for i in hod_qualifications]
#             print(hod_qualifications)
#             if hod_qualifications[0]:
#                 hod_data["Qualifications of HOD"] = {
#                     "Qualificaion": hod_qualifications[0],
#                     "Department": hod_qualifications[1],
#                     "Years of Experience": hod_qualifications[2]
#                 }
#                 hod_data["Contact Information of HOD"] = {
#                     "Phone Number": hod_qualifications[3],
#                     "Email": hod_qualifications[4]
#                 }
#                 hod_publications = response.css('div#prod-curriculum div.inner-box div.box p::text').getall()
#                 hod_publications = [clean_text(i) for i in hod_publications[13:]]
#                 hod_data["Publications of HOD"] = hod_publications
#                 mech_dep_data["Information on Head of Department of Mechanical Engineering Department"] = hod_data
#             else:
#                 hod_data = "None Specified"
#                 mech_dep_data["Information on Head of Department of Mechanical Engineering Department"] = hod_data

#             # Getting the faculty information of Mechanical department of CEK
#             faculty_cards = response.css('div.card.accordion.block')
#             faculty_info = []
#             for card in faculty_cards:
#                 name = card.css("div.names::text").get()
#                 phone_number = card.css("div.right_con::text").re_first(r"\d{10}")
#                 email = card.css("div.right_con::text").re_first(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%+-]+\.[a-zA-Z]{2,}")
#                 designation = card.css("div.col-lg-8 div.designation::text").get()
#                 if name:
#                     faculty_info.append(
#                         {
#                             "Name": name,
#                             "Designation": clean_text(designation),
#                             "Phone Number": phone_number,
#                             "Email": email
#                         }
#                     )
#             mech_dep_data["Mechanical Department Faculty Information"] = faculty_info
#             course_overview_data_cleaned ={
#                 "Department of Applied Science": mech_dep_data
#             }
#             self.total_department_data["About the departments of CEK"].append(course_overview_data_cleaned)
        
#         # Getting the information of Electrical Department of CEK
#         if(department_title == "Department of Electrical Engineering"):
#             eee_dep_data = {}
#             eee_dep_data["About the Electrical Engineering Department of CEK"] = ''.join([course_overview_data_cleaned[i] for i in range(4)])

#             # Getting the information on HOD of Electrical Engineering Department
#             hod_data = {}
#             hod_qualifications = response.css('div#prod-curriculum div.right_con::text').getall()
#             hod_qualifications = [clean_text(i) for i in hod_qualifications]
#             if hod_qualifications[0]:
#                 hod_data["Qualifications of HOD"] = {
#                     "Qualificaion": hod_qualifications[0],
#                     "Department": hod_qualifications[1],
#                     "Years of Experience": hod_qualifications[2]
#                 }
#                 hod_data["Contact Information of HOD"] = {
#                     "Phone Number": hod_qualifications[3],
#                     "Email": hod_qualifications[4]
#                 }
#                 hod_publications = response.css('div#prod-curriculum div.inner-box div.box p::text').getall()
#                 hod_publications = [clean_text(i) for i in hod_publications[13:]]
#                 hod_data["Publications of HOD"] = hod_publications
#                 eee_dep_data["Information on Head of Department of Electrical Engineering Department"] = hod_data
#             else:
#                 hod_data = "None Specified"
#                 eee_dep_data["Information on Head of Department of Electrical Engineering Department"] = hod_data

#             # Getting the faculty information of Mechanical department of CEK
#             faculty_cards = response.css('div.card.accordion.block')
#             faculty_info = []
#             for card in faculty_cards:
#                 name = card.css("div.names::text").get()
#                 phone_number = card.css("div.right_con::text").re_first(r"\d{10}")
#                 email = card.css("div.right_con::text").re_first(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%+-]+\.[a-zA-Z]{2,}")
#                 designation = card.css("div.col-lg-8 div.designation::text").get()
#                 if name:
#                     faculty_info.append(
#                         {
#                             "Name": name,
#                             "Designation": clean_text(designation),
#                             "Phone Number": phone_number,
#                             "Email": email
#                         }
#                     )
#             eee_dep_data["ELectrical Department Faculty Information"] = faculty_info
#             course_overview_data_cleaned ={
#                 "Department of Applied Science": eee_dep_data
#             }
#             self.total_department_data["About the departments of CEK"].append(course_overview_data_cleaned)

#         if(department_title == "Department of Applied Sciences"):
#             as_dep_data = {}
#             as_dep_data["About the Applied Science Department of CEK"] = course_overview_data_cleaned[0]

#             # Getting the information on HOD of Applied Science Department
#             hod_data = {}
#             hod_qualifications = response.css('div#prod-curriculum div.right_con::text').getall()
#             hod_qualifications = [clean_text(i) for i in hod_qualifications]
#             if hod_qualifications[0]:
#                 hod_data["Qualifications of HOD"] = {
#                     "Qualificaion": hod_qualifications[0],
#                     "Department": hod_qualifications[1],
#                     "Years of Experience": hod_qualifications[2]
#                 }
#                 hod_data["Contact Information of HOD"] = {
#                     "Phone Number": hod_qualifications[3],
#                     "Email": hod_qualifications[4]
#                 }
#                 hod_publications = response.css('div#prod-curriculum div.inner-box div.box p::text').getall()
#                 hod_publications = [clean_text(i) for i in hod_publications[13:]]
#                 hod_data["Publications of HOD"] = hod_publications
#                 as_dep_data["Information on Head of Department of Applied Science Department"] = hod_data
#             else:
#                 hod_data = "None Specified"
#                 as_dep_data["Information on Head of Department of Applied Science Department"] = hod_data

#             # Getting the faculty information of Applied Science department of CEK
#             faculty_cards = response.css('div.card.accordion.block')
#             faculty_info = []
#             for card in faculty_cards:
#                 name = card.css("div.names::text").get()
#                 phone_number = card.css("div.right_con::text").re_first(r"\d{10}")
#                 email = card.css("div.right_con::text").re_first(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%+-]+\.[a-zA-Z]{2,}")
#                 designation = card.css("div.col-lg-8 div.designation::text").get()
#                 if name:
#                     faculty_info.append(
#                         {
#                             "Name": name,
#                             "Designation": clean_text(designation),
#                             "Phone Number": phone_number,
#                             "Email": email
#                         }
#                     )
#             as_dep_data["Applied Science Faculty Information"] = faculty_info
#             course_overview_data_cleaned ={
#                 "Department of Applied Science": as_dep_data
#             } 
#             self.total_department_data["About the departments of CEK"].append(course_overview_data_cleaned)
        
    
#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_department_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_contact(scrapy.Spider):
#     name = 'contact'
#     # allowed_domains = ['cea.ac.in']
#     start_urls = ['https://www.ceknpy.ac.in/contact-us']

#     total_contact_data = {}

#     def parse(self, response):
#         data = []
#         contact_ = response.css('div.img-part.js-tilt p::text').getall()
#         # print(contact_)
#         transportation = {
#             "Way to reach College of Engineering Karungappally":{
#                 contact_[0]:contact_[1],
#                 contact_[2]:contact_[3],
#                 contact_[4]:contact_[5]
#             }
#         }
#         data.append(transportation)
#         phone_number = response.css('div.address-text span.des a::text').getall()
#         print(phone_number)
#         data.append({
#             "Contact number of CEK":phone_number
#         })
#         self.total_contact_data["Contact information of CEK"] = data
    
#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_contact_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_placement(scrapy.Spider):
#     name = 'placement_cell'
#     start_urls = ['https://www.ceknpy.ac.in/campus']

#     total_placement_data = {}

#     def parse(self,response):
#         data = []
#         desc = response.css('div.sec-title div.desc p::text').get()
#         data.append({
#             "About the Training and Placement Cell of CEK": desc
#         })
#         placement_result = response.css("table tbody tr td::text").getall()

#         for i in placement_result:
#             print(i)
#         count = 0
#         i = 3
#         placement_ls = []
#         while count < 60:
#             placement_ls.append({
#                 "SI":placement_result[i],
#                 "Name": placement_result[i+1],
#                 "Company Placed": placement_result[i+2]
#             })
#             count += 1
#             i+=3
#         data.append({
#             "List of Student who got placed":placement_ls
#         })
#         self.total_placement_data = data
    
#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_placement_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEK_library(scrapy.Spider):
#     name = 'Library'
#     start_urls = ['https://www.ceknpy.ac.in/library']

#     total_library_data = {}

#     def parse(self,response):
#         data = []
#         desc = response.css('div.sec-title div.desc h5::text').getall()

#         data.append({
#             "About the Library": desc[0],
#             "Library Timing": desc[2],
#             "Library Collection": desc[4],
#             "Journals and Magazines at the Library": desc[6],
#             "Membership of Library": desc[10]
#         })
#         self.total_library_data = {
#             "About the Library of CEK": data
#         }

#     def closed(self, response):
#         with open('college_json_data/cek.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_library_data)
#         with open('college_json_data/cek.json', 'w') as f:
#             json.dump(data,f,indent=4)

import scrapy
import json
import re
import scrapy.crawler
import os

def clean_text(item):
    item = re.sub(r'\\u[0-9a-fA-F]{4}', '', item)
    item = re.sub(r'[\u2013\u2019\u2022\u201a\u201c\u201d\u00b1\xa0]', '-', item)
    item = re.sub(r'\s+', ' ', item).strip()
    item = re.sub(r'<br>','',item)
    item = re.sub(r'-','',item).strip()
    return item

class CEK_principal(scrapy.Spider):
    name = 'principal'
    # allowed_domains = ['cea.ac.in']
    start_urls = ['https://ceknpy.ac.in/principal']

    total_principal_data = {}

    def parse(self, response):
        principal_data = {}
        principal_name = response.css('table tbody tr td p span strong::text').get()
        principal_other_data = response.css('table tbody tr td p span::text').getall()
        print(principal_name)
        print(principal_other_data[:5])
        principal_data["Principal Name"] = principal_name
        principal_data["Department of the principal"] = principal_other_data[0]
        principal_data["Address of the principal"] = {"college":principal_other_data[1],"place and PIN Code": principal_other_data[2]} 
        principal_data["Phone Number and Fax Number of the Principal"] = principal_other_data[3]
        principal_data["Email ID of the principal"] = principal_other_data[4]
        self.total_principal_data["Data about the Principal of CEK"] = principal_data

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_principal_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_principal_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_management(scrapy.Spider):
    name = 'board_members'
    # allowed_domains = ['cea.ac.in']
    start_urls = ['https://www.ceknpy.ac.in/management']

    total_management_data = {}

    def parse(self, response):
        member_data = response.css('div.sec-title div p span::text').getall()
        print(member_data)
        management_data = ' '.join(member_data)
        self.total_management_data["Data about Management of CEK"] = management_data

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_management_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_management_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_admin_staff(scrapy.Spider):
    name = 'admin_staff'
    # allowed_domains = ['cea.ac.in']
    start_urls = ['https://www.ceknpy.ac.in/adminstaff']

    total_admin_staff = {}

    def parse(self, response):
        final_admin_staff_data = []
        admin_staff_data = response.css('div.sec-title div table tr p::text').getall()
        cleaned_admin_staff_data = []
        for i in admin_staff_data:
            if i not in ['\xa0','\r\n']:
                cleaned_admin_staff_data.append(i)

        print(cleaned_admin_staff_data)
        i = 0
        while(i<len(cleaned_admin_staff_data)):
            data = {}
            data["Name"] = clean_text(cleaned_admin_staff_data[i])
            data["Designation"] = clean_text(cleaned_admin_staff_data[i+1])
            data["E-mail"] = clean_text(cleaned_admin_staff_data[i+2])
            data["Phone Number"] = clean_text(cleaned_admin_staff_data[i+3])
            i+=4
            final_admin_staff_data.append(data)
        self.total_admin_staff["Admnistration staff Data"] = final_admin_staff_data
        
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_admin_staff, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_admin_staff)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_overview(scrapy.Spider):
    name = 'overview'
    start_urls = ['https://www.ceknpy.ac.in/overview']

    total_overview_data = {}

    def parse(self, response):
        overview_data = response.css('div.sec-title div.desc p::text, div.sec-title div::text').getall()
        overview_data = [clean_text(text.replace('\r\n', '')) for text in overview_data]

        # Basic college information
        self.total_overview_data['college_overview'] = clean_text(overview_data[1] + ' ' + overview_data[2])
        self.total_overview_data['college_vision'] = overview_data[3]
        self.total_overview_data['college_mission'] = overview_data[4]

        # B.Tech programs
        for i, program in enumerate(overview_data[5:9], 1):
            self.total_overview_data[f'btech_program_{i}'] = program

        # M.Tech program
        self.total_overview_data['mtech_programs'] = overview_data[10]

        # Other college features and facilities
        feature_mappings = {
            'memorandum_of_understanding': overview_data[11].replace('\r', ''),
            'student_professional_bodies': overview_data[12],
            'library_details': overview_data[13],
            'training_placement_cell': overview_data[14],
            'alumni_details': overview_data[15],
            'parent_teacher_association': overview_data[16],
            'hostel_facilities': overview_data[17],
            'canteen_facilities': overview_data[18],
            'atm_facilities': overview_data[19],
            'sports_cultural_festivals': overview_data[20]
        }

        # Add all features to the flattened structure
        for key, value in feature_mappings.items():
            self.total_overview_data[key] = value

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_overview_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_overview_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_infrastructure(scrapy.Spider):
    name = 'infrastructure'
    start_urls = ['https://www.ceknpy.ac.in/infrastructure']

    total_infrastructure_data = {}

    def parse(self, response):
        data = {}
        infra_data = response.css('div.sec-title div.desc p::text').getall()
        for i in infra_data:
            i.replace('\r\n','')

        data["Data about building and Labs"] = clean_text(infra_data[0])
        data["Information about the roads to the college (CEK)"] = infra_data[1]
        data["About the placement activities"] = infra_data[2]
        data["PTA (parent teacher association) at CEK"] = infra_data[3]
        data["Transportation details at CEK"] = infra_data[4]
        data["Water supply at CEK"] = infra_data[5]
        self.total_infrastructure_data["About the Infrastructure at CEK"] = data

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_infrastructure_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_infrastructure_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)
        
class CEK_anti_ragging_squad(scrapy.Spider):
    name = 'Anti_Ragging_Squad'
    start_urls = ['https://www.ceknpy.ac.in/antiragging']

    total_Anti_Ragging_Squad = {}

    def parse(self, response):
        ars_data = response.css('table tr td p span::text').getall()
        squad_list = ars_data[13:]  # Get squad members data
        
        # Process squad members in pairs (name and phone)
        for i in range(0, len(squad_list), 2):
            member_number = (i // 2) + 1  # Calculate member number
            
            # Add member data with numbered keys
            self.total_Anti_Ragging_Squad[f'anti_ragging_member_{member_number}_name'] = squad_list[i].strip()
            if i + 1 < len(squad_list):  # Check if phone number exists
                self.total_Anti_Ragging_Squad[f'anti_ragging_member_{member_number}_phone'] = squad_list[i+1].strip()
    
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_Anti_Ragging_Squad, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_Anti_Ragging_Squad)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_Departmentdata(scrapy.Spider):
    name = 'department'
    start_urls = ['https://www.ceknpy.ac.in/departments/']

    total_department_data = {}

    def parse(self, response):
        department_links = response.css('div.content-part span a::attr(href)').getall()
        for department_link in department_links:
            yield response.follow(department_link, self.parse_department)

    def parse_department(self, response):
        department_title = response.css('div.breadcrumbs-text ul li::text').getall()[-1].strip()
        dept_code = self._get_department_code(department_title)
        
        # Department Overview
        course_overview_data = response.css('div.course-overview div p::text, div.course-overview div h3::text').getall()
        overview_data = [clean_text(i) for i in course_overview_data if clean_text(i)]
        
        # Basic department info
        # self.total_department_data[f'dept_{dept_code}_name'] = department_title
        self.total_department_data[f'dept_{dept_code}_overview'] = ' '.join(overview_data[:5])

        # Special handling for CS department seats
        if dept_code == 'computer science':
            self.total_department_data['dept_cs_btech_seats'] = "120 Seats"
            self.total_department_data['dept_cs_ai_ds_seats'] = "60 Seats"
            
            # Program outcomes for CS
            # for i, outcome in enumerate(overview_data[5:35], 1):
            #     self.total_department_data[f'dept_cs_program_outcome_{i}'] = outcome

        # HOD Information
        hod_qualifications = response.css('div#prod-curriculum div.right_con::text').getall()
        hod_qualifications = [clean_text(i) for i in hod_qualifications]
        
        if hod_qualifications and hod_qualifications[0]:
            self.total_department_data[f'dept_{dept_code}_hod_name'] = response.css("div.names::text").get().strip()
            # self.total_department_data[f'dept_{dept_code}_hod_qualification'] = hod_qualifications[0]
            # self.total_department_data[f'dept_{dept_code}_hod_department'] = hod_qualifications[1]
            # self.total_department_data[f'dept_{dept_code}_hod_experience'] = hod_qualifications[2]
            self.total_department_data[f'dept_{dept_code}_hod_phone'] = hod_qualifications[3]
            self.total_department_data[f'dept_{dept_code}_hod_email'] = hod_qualifications[4]
            
            # HOD Publications
            hod_publications = response.css('div#prod-curriculum div.inner-box div.box p::text').getall()
            for i, pub in enumerate([clean_text(p) for p in hod_publications[13:]], 1):
                self.total_department_data[f'dept_{dept_code}_hod_publication_{i}'] = pub

        # Faculty Information
        faculty_cards = response.css('div.card.accordion.block')
        for i, card in enumerate(faculty_cards, 1):
            name = card.css("div.names::text").get()
            if name:
                faculty_base_key = f'dept_{dept_code}_faculty_{i}'
                self.total_department_data[f'{faculty_base_key}_name'] = name.strip()
                # self.total_department_data[f'{faculty_base_key}_designation'] = clean_text(card.css("div.col-lg-8 div.designation::text").get())
                self.total_department_data[f'{faculty_base_key}_phone'] = card.css("div.right_con::text").re_first(r"\d{10}")
                self.total_department_data[f'{faculty_base_key}_email'] = card.css("div.right_con::text").re_first(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%+-]+\.[a-zA-Z]{2,}")

    def _get_department_code(self, department_title):
        """Helper function to get department code from title"""
        if "Computer Science" in department_title:
            return "computer science"
        elif "Electronics" in department_title:
            return "Electronics"
        elif "Mechanical" in department_title:
            return "Mechanical"
        elif "Electrical" in department_title:
            return "Electrical"
        elif "Applied Sciences" in department_title:
            return "Applied Sciences"
        return "unknown"

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_department_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_department_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_contact(scrapy.Spider):
    name = 'contact'
    start_urls = ['https://www.ceknpy.ac.in/contact-us']

    total_contact_data = {}

    def parse(self, response):
        contact_ = response.css('div.img-part.js-tilt p::text').getall()
        
        # Flatten transportation data
        for i in range(0, len(contact_), 2):
            if i + 1 < len(contact_):
                self.total_contact_data[f'transportation_method_{i//2 + 1}_type'] = clean_text(contact_[i])
                self.total_contact_data[f'transportation_method_{i//2 + 1}_details'] = clean_text(contact_[i+1])

        # Flatten phone numbers
        phone_numbers = response.css('div.address-text span.des a::text').getall()
        for i, phone in enumerate(phone_numbers, 1):
            self.total_contact_data[f'contact_phone_{i}'] = clean_text(phone)

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_contact_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_contact_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_placement(scrapy.Spider):
    name = 'placement_cell'
    start_urls = ['https://www.ceknpy.ac.in/campus']

    total_placement_data = {}

    def parse(self, response):
        # Get placement cell description
        desc = response.css('div.sec-title div.desc p::text').get()
        self.total_placement_data['placement_cell_description'] = clean_text(desc)

        # Flatten placement results
        # placement_result = response.css("table tbody tr td::text").getall()
        
        # i = 3  # Starting index for placement data
        # placement_count = 1
        # while placement_count <= 60 and i + 2 < len(placement_result):
        #     self.total_placement_data[f'placed_student_{placement_count}_si_no'] = clean_text(placement_result[i])
        #     self.total_placement_data[f'placed_student_{placement_count}_name'] = clean_text(placement_result[i+1])
        #     self.total_placement_data[f'placed_student_{placement_count}_company'] = clean_text(placement_result[i+2])
        #     placement_count += 1
        #     i += 3

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_placement_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_placement_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEK_library(scrapy.Spider):
    name = 'Library'
    start_urls = ['https://www.ceknpy.ac.in/library']

    total_library_data = {}

    def parse(self, response):
        desc = response.css('div.sec-title div.desc h5::text').getall()
        
        # Map indices to meaningful keys
        library_mappings = {
            0: 'library_description',
            2: 'library_timing',
            4: 'library_collection',
            6: 'library_journals_magazines',
            10: 'library_membership'
        }
        
        # Flatten library data
        for index, key in library_mappings.items():
            if index < len(desc):
                self.total_library_data[key] = clean_text(desc[index])

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_library_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cek.json'):
            try:
                with open('college_json_data/cek.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_library_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(existing_data, f, indent=4)


class CEK_fees_structure(scrapy.Spider):
    name = 'fees structure'
    start_urls = ['https://www.ceknpy.ac.in/adm2024']

    total_fee_data = {}

    def parse(self,response):
        data = {
            "Academic Eligibility, Seats Distribution and Fees Structure of College of Engineering Karunagapally": {
                "Information about admission of Merit Lower Fee Seats of CEK": "Allotment is through the Commissioner of Entrance Examinations, Govt. of Kerala, from the KEAM rank list. No of seats : 60 seats  in Computer Science, 30 seats  in AD, 22 seats in Electronics, 22 seats in Mechanical Engineering. The Fees for this category is Rs 20,000. The academic Eligibility of this group is: Candidates who have passed Higher Secondary Examination, Kerala, or Examinations recognized as equivalent thereto, with Physics and Mathematics as compulsory subjects and Chemistry as one of the optional subjects with atleast 45% marks put together in the above subjects are eligible for admission. In case, the candidate has not studied Chemistry, the marks obtained in Computer Science shall be considered. In case, the candidate has not studied Chemistry and Computer Science the marks obtained in Biotechnology shall be considered. In case, the candidate has not studied Chemistry, Computer Science and Biotechnology the marks obtained in Biology shall be considered. The marks as shown in the mark list of the Board of Examination obtained from the respective Higher Secondary Board shall be considered for academic eligibility. The Scheduled Castes and Scheduled Tribes candidates (SC/ST/ SEBC or PwD) the minimum mark obtained in Physics, Mathematics, Chemistry/Computer Science/Biotechnology/Biology taken together in qualifying examination shall be 40% instead of 45%.",
                "Information about the Merit Full seats":" The allotment of students is By the Commissioner of Entrance Examinations, Govt. of Kerala, from the KEAM rank list. The seat distribution is: 54 seats in CS, 27 seats in AD, 6 seats in EC, 6 seats in ME. The Fees structure of this category is: Rs. 65,000/- per year. Can pay in two installments (35,000 at the beginning of odd semester + 30,000 at the beginning of even semester)",
                "Information about NRI Seats": "The allotment is By the Director of IHRD from the list of applicants. No of seats is as follows: 2 seats each in ECE and ME,  6 seats in CS, 3 seats in AD. The Academic Eligibility is as follows:  Candidates who have passed Higher Secondary Examination, Kerala or Examinations recognized as equivalent thereto, with 45% marks in Mathematics, Physics & Chemistry put together (or equivalent subjects specified in prospectus of KEAM-2023) are eligible for admission under NRI seats in Engineering Colleges under IHRD. The Scheduled Castes and Scheduled Tribes candidates need only a pass in the qualifying examination. The Fees Structure for NRI Seats are as follows: Fees : Rs. 1,00,000/- per year. A refundable deposit of Rs.1,25,000/- to be paid at the time of admission ( as was followed on the previous year, till any modification ordered by Govt. of Kerala / IHRD)"            
            },
            "Course or Programmes Offered at College of Engineering Karungapally": [
                "Computer Science & Engineering (CS) (NBA Accredited)",
                "Electronics & Communication Engineering (EC) ",
                "Mechanical Engineering (ME)",
                "B-Tech  with Artificial  Intelligence and Data science",
                "Bachelor of Computer Application(BCA)",
                "Bachelor of Business Administration(BBA)"

            ]
        }
        self.total_fee_data = data

    def closed(self, response):
        with open('college_json_data/cek.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_fee_data)
        with open('college_json_data/cek.json', 'w') as f:
            json.dump(data,f,indent=4)