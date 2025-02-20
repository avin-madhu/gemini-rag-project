# import scrapy
# import json
# import os
# import re
# from scrapy.crawler import CrawlerProcess
# from scrapy.utils.project import get_project_settings

# def clean_text(item):
#     item = re.sub(r'\\u[0-9a-fA-F]{4}', '', item)
#     item = re.sub(r'[\u2013\u2019\u2022\u201a\u201c\u201d\u00b1\xa0]', '-', item)
#     item = re.sub(r'\s+', ' ', item).strip()
#     item = re.sub(r'<br>','',item)
#     item = re.sub(r'-','',item).strip()
#     return item

# class aboutSpider(scrapy.Spider):
#     name = "about"
#     start_urls = ["https://cep.ac.in/about/aboutUs"]

#     def parse(self, response):
#         college_data = {
#             # "title": response.css("h1::text").get(),
#             "introduction of College of Engineering Poonjar": response.css("p.w-full::text").get(),
#             # "vision_title": response.css("h1:contains('Vision')::text").get(),
#             "vision description of College of Engineering Poonjar": response.css("div:contains('Scientific advancement')::text").get(),
#             # "mission_title": response.css("h1:contains('Mission')::text").get(),
#             "mission_point_1 of College of Engineering Poonjar": response.css("div ul li:nth-child(1)::text").get(),
#             "mission_point_2 of College of Engineering Poonjar": response.css("div ul li:nth-child(2)::text").get(),
#             "mission_point_3 of College of Engineering Poonjar": response.css("div ul li:nth-child(3)::text").get(),
#             # "document_text": response.css("a[target='_blank'] div p::text").get(),
#             # "document_link": response.urljoin(response.css("a[target='_blank']::attr(href)").get()),
#         }

#         file_path = "college_json_data/cep.json"

#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 data = json.load(file)
#         else:
#             data = []
#         data.update(college_data)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(data, file, ensure_ascii=False, indent=4)

#         self.log(f"Data has been appended to {file_path}")

# class PrincipalDetailsSpider(scrapy.Spider):
#     name = "principal_details"
#     start_urls = ["https://cep.ac.in/about/principal"]

#     def parse(self, response):
#         principal_details = {
#             "principal_name of College of Engineering Poonjar": response.css("span[class*='text-']::text").get(),
#             # "designation": response.css("span.block.font-medium::text").get(),
#             "image_url of principal of College of Engineering Poonjar": response.css("img[class*='w-']::attr(src)").get(),
#             "message from principal of College of Engineering Poonjar": response.css("p.text-justify::text").get(),
#             "email of principal of College of Engineering Poonjar": response.css("td:contains('Email :') + td::text").get(),
#             "phone of principal of College of Engineering Poonjar": response.css("td:contains('Phone :') + td::text").get(),
#         }

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}


#         else:
#             existing_data = {}
#         existing_data.update(principal_details)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Principal details have been saved to {file_path}")

# class ContactDetailsSpider(scrapy.Spider):
#     name = "contact_details"
#     start_urls = ["https://cep.ac.in/about/contact"]

#     def parse(self, response):
#         address = [line.get() for line in response.css("h3:contains('Address') + p::text, h3:contains('Address') + p ~ p::text")[:4]]
#         phone_numbers = [phone.get() for phone in response.css("h3:contains('Phone') + p::text, h3:contains('Phone') + p + p::text")]
#         email_addresses = [email.get() for email in response.css("h3:contains('Email') + p::text, h3:contains('Email') + p ~ p::text")]
#         contact_details = {
#             "principal_name of College of Engineering Poonjar": response.css("h3:contains('Principal') + p::text").get(),
#             "principal_phone of College of Engineering Poonjar": response.css("h3:contains('Principal') + p + p::text").get().replace("Phone: ", ""),
            
#             "placement_officer_name of College of Engineering Poonjar": response.css("h3:contains('Placement Officer') + p::text").get(),
#             "placement_officer_phone of College of Engineering Poonjar": response.css("h3:contains('Placement Officer') + p + p::text").get().replace("Phone: ", ""),
#             "placement_officer_email of College of Engineering Poonjar": response.css("h3:contains('Placement Officer') + p + p + p::text").get().replace("Email: ", ""),
            
#             # "address of College of Engineering Poonjar": [
#             #     line.get() for line in response.css("h3:contains('Address') + p::text, h3:contains('Address') + p ~ p::text")[:4]
#             # ],
            
#             # "phone_numbers of College of Engineering Poonjar": [
#             #     phone.get() for phone in response.css("h3:contains('Phone') + p::text, h3:contains('Phone') + p + p::text")
#             # ],
            
#             # "email_addresses of College of Engineering Poonjar": [
#             #     email.get() for email in response.css("h3:contains('Email') + p::text, h3:contains('Email') + p ~ p::text")
#             # ]
#             "address of College of Engineering Poonjar": " ".join(address),
#             "phone_numbers of College of Engineering Poonjar": ", ".join(phone_numbers),
#             "email_addresses of College of Engineering Poonjar": ", ".join(email_addresses)
#         }

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}
            
#         existing_data.update(contact_details)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Contact details have been saved to {file_path}")

# class ComputerScienceDepartmentSpider(scrapy.Spider):
#     name = "computer_science_department"
#     start_urls = ["https://cep.ac.in/departments/cs"]

#     def parse(self, response):
#         department_name = response.css("h1.text-center::text").get().strip()
#         department_introduction = response.css("div.text-justify::text").get().strip()
#         department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


#         department_mission = [
#         mission.strip() for mission in response.css("div.grid div:nth-child(2) ul li::text").getall()
#     ]


#         faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
#         faculty_flat = [
#             {
#                 "faculty_name": faculty.css("span.text-dark::text").get().strip(),
#                 "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
#                 "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
#                 "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
#                 "faculty_profile_link": faculty.css("a::attr(href)").get()
#             }
#             for faculty in faculty_details
#         ]
#         lab_staff_details = faculty_details[-5:]
#         lab_staff_flat = [
#             {
#                 "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
#                 "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
#                 "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
#                 "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
#             }
#             for staff in lab_staff_details
#         ]

#         flattened_data = {
#             "department_name of Computer Science department": department_name,
#             "department_introduction of Computer Science department": department_introduction,
#             "department_vision of Computer Science department": department_vision,
#             "department_mission of Computer Science department": "; ".join(department_mission), 
#         }

#         for i, faculty in enumerate(faculty_flat, start=1):
#             flattened_data.update({
#                 f"faculty_{i}_name of Computer Science department": faculty["faculty_name"],
#                 f"faculty_{i}_designation of Computer Science department": faculty["faculty_designation"],
#                 f"faculty_{i}_qualification of Computer Science department": faculty["faculty_qualification"],
#                 f"faculty_{i}_image_url of Computer Science department": faculty["faculty_image_url"],
#                 f"faculty_{i}_profile_link of Computer Science department": faculty["faculty_profile_link"]
#             })

#         # Add lab staff details to flattened data
#         for i, staff in enumerate(lab_staff_flat, start=1):
#             flattened_data.update({
#                 f"lab_staff_{i}_name of Computer Science department": staff["lab_staff_name"],
#                 f"lab_staff_{i}_designation of Computer Science department": staff["lab_staff_designation"],
#                 f"lab_staff_{i}_qualification of Computer Science department": staff["lab_staff_qualification"],
#                 f"lab_staff_{i}_image_url of Computer Science department": staff["lab_staff_image_url"]
#             })

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}
#         existing_data.update(flattened_data)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Computer Science department details have been saved to {file_path}")

# class ELectricalElectronicsDepartmentSpider(scrapy.Spider):
#     name = "electrical_electronics_department"
#     start_urls = ["https://cep.ac.in/departments/ee"]

#     def parse(self, response):
#         # department_name = response.css("h1.text-center::text").get().strip()
#         department_introduction = response.css("div.text-justify::text").get().strip()
#         department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


#         department_mission = [
#         mission.strip() for mission in response.css("div img[src*='mission.png'] + h1 + div ul.list-disc li::text").getall()
#     ]


#         faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
#         faculty_flat = [
#             {
#                 "faculty_name": faculty.css("span.text-dark::text").get().strip(),
#                 "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
#                 "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
#                 "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
#                 "faculty_profile_link": faculty.css("a::attr(href)").get()
#             }
#             for faculty in faculty_details
#         ]
#         lab_staff_details = faculty_details[-5:]
#         lab_staff_flat = [
#             {
#                 "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
#                 "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
#                 "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
#                 "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
#             }
#             for staff in lab_staff_details
#         ]

#         flattened_data = {
#             # "department_name of ELectrical and ELectronics department": department_name,
#             "department_introduction of ELectrical and ELectronics department": department_introduction,
#             "department_vision of ELectrical and ELectronics department": department_vision,
#             "department_mission of ELectrical and ELectronics department": "; ".join(department_mission), 
#         }

#         for i, faculty in enumerate(faculty_flat, start=1):
#             flattened_data.update({
#                 f"faculty_{i}_name of ELectrical and ELectronics department": faculty["faculty_name"],
#                 f"faculty_{i}_designation of ELectrical and ELectronics department": faculty["faculty_designation"],
#                 f"faculty_{i}_qualification of ELectrical and ELectronics department": faculty["faculty_qualification"],
#                 f"faculty_{i}_image_url of ELectrical and ELectronics department": faculty["faculty_image_url"],
#                 f"faculty_{i}_profile_link of ELectrical and ELectronics department": faculty["faculty_profile_link"]
#             })

#         # Add lab staff details to flattened data
#         for i, staff in enumerate(lab_staff_flat, start=1):
#             flattened_data.update({
#                 f"lab_staff_{i}_name of ELectrical and ELectronics department": staff["lab_staff_name"],
#                 f"lab_staff_{i}_designation of ELectrical and ELectronics department": staff["lab_staff_designation"],
#                 f"lab_staff_{i}_qualification of ELectrical and ELectronics department": staff["lab_staff_qualification"],
#                 f"lab_staff_{i}_image_url of ELectrical and ELectronics department": staff["lab_staff_image_url"]
#             })

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}
#         existing_data.update(flattened_data)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"ELectrical and ELectronics department details have been saved to {file_path}")


# class Electronics_CommunicationDepartmentSpider(scrapy.Spider):
#     name = "electronics_communication_department"
#     start_urls = ["https://cep.ac.in/departments/ec"]

#     def parse(self, response):
#         # department_name = response.css("h1.text-center::text").get().strip()
#         department_introduction = response.css("div.text-justify::text").get().strip()
#         department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


#         department_mission = [
#         mission.strip() for mission in response.css("div img[src*='mission.png'] + h1 + div ul.list-disc li::text").getall()
#     ]


#         faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
#         faculty_flat = [
#             {
#                 "faculty_name": faculty.css("span.text-dark::text").get().strip(),
#                 "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
#                 "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
#                 "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
#                 "faculty_profile_link": faculty.css("a::attr(href)").get()
#             }
#             for faculty in faculty_details
#         ]
#         lab_staff_details = faculty_details[-5:]
#         lab_staff_flat = [
#             {
#                 "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
#                 "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
#                 "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
#                 "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
#             }
#             for staff in lab_staff_details
#         ]

#         flattened_data = {
#             # "department_name of ELectrical and ELectronics department": department_name,
#             "department_introduction of Electronics and communication department": department_introduction,
#             "department_vision of Electronics and communication department": department_vision,
#             "department_mission of Electronics and communication department": "; ".join(department_mission), 
#         }

#         for i, faculty in enumerate(faculty_flat, start=1):
#             flattened_data.update({
#                 f"faculty_{i}_name of Electronics and communication department": faculty["faculty_name"],
#                 f"faculty_{i}_designation of Electronics and communication department": faculty["faculty_designation"],
#                 f"faculty_{i}_qualification of Electronics and communication department": faculty["faculty_qualification"],
#                 f"faculty_{i}_image_url of Electronics and communication department": faculty["faculty_image_url"],
#                 f"faculty_{i}_profile_link of Electronics and communication department": faculty["faculty_profile_link"]
#             })

#         # Add lab staff details to flattened data
#         for i, staff in enumerate(lab_staff_flat, start=1):
#             flattened_data.update({
#                 f"lab_staff_{i}_name of Electronics and communication department": staff["lab_staff_name"],
#                 f"lab_staff_{i}_designation of Electronics and communication department": staff["lab_staff_designation"],
#                 f"lab_staff_{i}_qualification of Electronics and communication department": staff["lab_staff_qualification"],
#                 f"lab_staff_{i}_image_url of Electronics and communication department": staff["lab_staff_image_url"]
#             })

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}
#         existing_data.update(flattened_data)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Electronics and communication department details have been saved to {file_path}")


# class AppliedScienceDepartmentSpider(scrapy.Spider):
#     name = "applied_science_department"
#     start_urls = ["https://cep.ac.in/departments/sah"]

#     def parse(self, response):
#         # department_name = response.css("h1.text-center::text").get().strip()
#         department_introduction = response.css("div.text-justify::text").get().strip()
#         department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


#         department_mission = [
#         mission.strip() for mission in response.css("div img[src*='mission.png'] + h1 + div ul.list-disc li::text").getall()
#     ]


#         faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
#         faculty_flat = [
#             {
#                 "faculty_name": faculty.css("span.text-dark::text").get().strip(),
#                 "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
#                 "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
#                 "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
#                 "faculty_profile_link": faculty.css("a::attr(href)").get()
#             }
#             for faculty in faculty_details
#         ]
#         lab_staff_details = faculty_details[-5:]
#         lab_staff_flat = [
#             {
#                 "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
#                 "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
#                 "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
#                 "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
#             }
#             for staff in lab_staff_details
#         ]

#         flattened_data = {
#             # "department_name of ELectrical and ELectronics department": department_name,
#             "department_introduction of Applied Science department": department_introduction,
#             "department_vision of Applied Science department": department_vision,
#             "department_mission of Applied Science department": "; ".join(department_mission), 
#         }

#         for i, faculty in enumerate(faculty_flat, start=1):
#             flattened_data.update({
#                 f"faculty_{i}_name of Applied Science department": faculty["faculty_name"],
#                 f"faculty_{i}_designation of Applied Science department": faculty["faculty_designation"],
#                 f"faculty_{i}_qualification of Applied Science department": faculty["faculty_qualification"],
#                 f"faculty_{i}_image_url of Applied Science department": faculty["faculty_image_url"],
#                 f"faculty_{i}_profile_link of Applied Science department": faculty["faculty_profile_link"]
#             })

#         # Add lab staff details to flattened data
#         for i, staff in enumerate(lab_staff_flat, start=1):
#             flattened_data.update({
#                 f"lab_staff_{i}_name of Applied Science department": staff["lab_staff_name"],
#                 f"lab_staff_{i}_designation of Applied Science department": staff["lab_staff_designation"],
#                 f"lab_staff_{i}_qualification of Applied Science department": staff["lab_staff_qualification"],
#                 f"lab_staff_{i}_image_url of Applied Science department": staff["lab_staff_image_url"]
#             })

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}
#         existing_data.update(flattened_data)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Applied Science department details have been saved to {file_path}")        


# class IQACSpider(scrapy.Spider):
#     name = "iqac_spider"
#     start_urls = ["https://cep.ac.in/iqac"] 

#     def parse(self, response):
#         about_iqac = response.xpath("//h2[contains(text(), 'About IQAC')]/following-sibling::p/text()").get()

#         members = response.xpath("//table[contains(@class, 'min-w-full')]/tbody/tr")
#         member_data = []
        
#         # Collect title and name pairs
#         for i in range(0, len(members), 2):
#             title = members[i].xpath("./td[@colspan='2']/text()").get()
#             name = members[i+1].xpath("./td[@class='px-6 py-4 text-sm font-medium text-gray-800 dark:text-gray-200 text-center']/text()").get()

#             if title and name:
#                 member_data.append({
#                     "title": title.strip() if title else None,
#                     "name": name.strip() if name else None
#                 })

#         flattened_data = {
#             "about_iqac": about_iqac.strip() if about_iqac else None,
#         }
#         for i, member in enumerate(member_data, start=1):
#             flattened_data.update({
#                 f"IQAC {member['title']} member_{i}_name": member["name"]
#             })

#         file_path = "college_json_data/cep.json"

#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}

#         existing_data.update(flattened_data)

#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"IQAC department details have been saved to {file_path}")

# class AcademicCouncilSpider(scrapy.Spider):
#     name = "academic_council"
#     start_urls = ["https://cep.ac.in/council"] 

#     # def parse(self, response):

#     #     academic_council_intro = response.xpath("//div[@class='max-w-[1200px] mx-auto']/p/text()").get()

#     #     functions = response.xpath("//h2[contains(text(), 'The functions of the Academic Council are as below:')]/following-sibling::ul[1]/li/text()").getall()
#     #     flattened_data = {
#     #         "academic_council_intro": academic_council_intro.strip() if academic_council_intro else None,
#     #         "academic_council_functions": functions if functions else [],
#     #     }
#     def parse(self, response):
#         academic_council_intro = response.xpath("//div[@class='max-w-[1200px] mx-auto']/p/text()").get()

#         functions = response.xpath("//h2[contains(text(), 'The functions of the Academic Council are as below:')]/following-sibling::ul[1]/li/text()").getall()

#         flattened_functions = {
#             f"function_{i+1}_of_academic_council": function.strip()
#             for i, function in enumerate(functions)
#         }

#         flattened_data = {
#             "academic_council_intro": academic_council_intro.strip() if academic_council_intro else None,
#         }

#         flattened_data.update(flattened_functions)
#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}

#         existing_data.update(flattened_data)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Academic Council details have been saved to {file_path}")        


# class ProgramsOfferedSpider(scrapy.Spider):
#     name = "programs_offered"
#     start_urls = ["https://cep.ac.in/programs"]

#     def parse(self, response):

#         post_graduate_courses = response.xpath("//h2[contains(text(), 'Post Graduate')]/following-sibling::div//table/tbody/tr")
#         post_graduate_data = {}
#         for i, course in enumerate(post_graduate_courses):
#             course_name = course.xpath(".//td[1]/text()").get().strip()
#             intake_number = course.xpath(".//td[2]/text()").get().strip()
#             # post_graduate_data[f"pg_course_{i+1}_name"] = course_name
#             post_graduate_data[f"pg_course_{i+1} {course_name}_intake"] = intake_number

#         under_graduate_courses = response.xpath("//h2[contains(text(), 'Under Graduate')]/following-sibling::div//table/tbody/tr")
#         under_graduate_data = {}
#         for i, course in enumerate(under_graduate_courses):
#             course_name = course.xpath(".//td[1]/text()").get().strip()
#             intake_number = course.xpath(".//td[2]/text()").get().strip()
#             # under_graduate_data[f"ug_course_{i+1}_name"] = course_name
#             under_graduate_data[f"ug_course_{i+1} {course_name}_intake"] = intake_number

#         diploma_courses = response.xpath("//h2[contains(text(), 'Diploma')]/following-sibling::div//table/tbody/tr")
#         diploma_data = {}
#         for i, course in enumerate(diploma_courses):
#             course_name = course.xpath(".//td[1]/text()").get().strip()
#             intake_number = course.xpath(".//td[2]/text()").get().strip()
#             # diploma_data[f"diploma_course_{i+1}_name"] = course_name
#             diploma_data[f"diploma_course_{i+1} {course_name}_intake"] = intake_number

#         flattened_data = {
#             **post_graduate_data,
#             **under_graduate_data,
#             **diploma_data
#         }

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}

#         existing_data.update(flattened_data)

#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Programs offered details have been saved to {file_path}")

# class AdmissionDetailsSpider(scrapy.Spider):
#     name = "admission_details"
#     start_urls = ["https://cep.ac.in/admission"]  

#     def parse(self, response):
#         btech_admission_data = {
#             "btech_academic_eligibility": response.xpath("//h2[contains(text(), 'B.Tech Admission')]/following-sibling::p[1]//text()").get().strip(),
#             "btech_allotment": response.xpath("//h2[contains(text(), 'B.Tech Admission')]/following-sibling::p[2]//text()").get().strip(),
#             "btech_exam_link": response.xpath("//h2[contains(text(), 'B.Tech Admission')]/following-sibling::div[1]//a/@href").get(),
#         }

#         diploma_admission_data = {
#             "diploma_academic_eligibility": response.xpath("//h2[contains(text(), 'Diploma Admission')]/following-sibling::p[1]//text()").get().strip(),
#             "diploma_allotment": response.xpath("//h2[contains(text(), 'Diploma Admission')]/following-sibling::p[2]//text()").get().strip(),
#             "diploma_exam_link": response.xpath("//h2[contains(text(), 'Diploma Admission')]/following-sibling::div[1]//a/@href").get(),
#         }

#         mca_admission_data = {
#             "mca_academic_eligibility": response.xpath("//h2[contains(text(), 'MCA Admission')]/following-sibling::p[1]//text()").get().strip(),
#             "mca_allotment": response.xpath("//h2[contains(text(), 'MCA Admission')]/following-sibling::p[2]//text()").get().strip(),
#             "mca_exam_link": response.xpath("//h2[contains(text(), 'MCA Admission')]/following-sibling::div[1]//a/@href").get(),
#         }

#         lateral_entry_admission_data = {
#             "lateral_entry_academic_eligibility": response.xpath("//h2[contains(text(), 'B.Tech Lateral Entry Admission')]/following-sibling::p[1]//text()").get().strip(),
#             "lateral_entry_allotment": response.xpath("//h2[contains(text(), 'B.Tech Lateral Entry Admission')]/following-sibling::p[2]//text()").get().strip(),
#             "lateral_entry_exam_link": response.xpath("//h2[contains(text(), 'B.Tech Lateral Entry Admission')]/following-sibling::div[1]//a/@href").get(),
#         }

#         contact_data = {}
#         contact_names = [
#             response.xpath("//table/tbody/tr[1]/td[1]//text()").get().strip(),
#             response.xpath("//table/tbody/tr[2]/td[1]//text()").get().strip(),
#             response.xpath("//table/tbody/tr[3]/td[1]//text()").get().strip(),
#         ]
#         contact_numbers = [
#             response.xpath("//table/tbody/tr[1]/td[2]//text()").get().strip(),
#             response.xpath("//table/tbody/tr[2]/td[2]//text()").get().strip(),
#             response.xpath("//table/tbody/tr[3]/td[2]//text()").get().strip(),
#         ]

#         for i, (name, number) in enumerate(zip(contact_names, contact_numbers), start=1):
#             contact_data[f"contact_{i}_name"] = name
#             contact_data[f"{name}_number"] = number

#         fee_structure_data = {
#             "mca_fee_link": response.xpath("//table[contains(@class, 'min-w-full')]//tr[1]/td[2]//a/@href").get(),
#             "btech_fee_link": response.xpath("//table[contains(@class, 'min-w-full')]//tr[2]/td[2]//a/@href").get(),
#             "diploma_fee_link": response.xpath("//table[contains(@class, 'min-w-full')]//tr[3]/td[2]//a/@href").get(),
#         }

#         flattened_data = {
#             **btech_admission_data,
#             **diploma_admission_data,
#             **mca_admission_data,
#             **lateral_entry_admission_data,
#             **contact_data,
#             **fee_structure_data,
#         }

#         file_path = "college_json_data/cep.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r", encoding="utf-8") as file:
#                 try:
#                     existing_data = json.load(file)
#                     if not isinstance(existing_data, dict):
#                         self.log("Existing file is not a dictionary. Replacing content.")
#                         existing_data = {}
#                 except json.JSONDecodeError:
#                     self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
#                     existing_data = {}
#         else:
#             existing_data = {}

#         existing_data.update(flattened_data)
#         with open(file_path, "w", encoding="utf-8") as file:
#             json.dump(existing_data, file, ensure_ascii=False, indent=4)

#         self.log(f"Admission details have been saved to {file_path}")             

# class CEP_facilities(scrapy.Spider):
#     name = 'facilities'
#     start_urls = [
#         'https://cep.ac.in/facilities/computer',
#         'https://cep.ac.in/facilities/library',
#         'https://cep.ac.in/facilities/seminar',
#         'https://cep.ac.in/facilities/transport',
#         'https://cep.ac.in/facilities/hostel',
#         'https://cep.ac.in/facilities/canteen'
#     ]

#     facilities_data = {}
#     total_fac = {}

#     def parse(self, response):
#         facility_name = response.url.split('/')[-1]
#         if facility_name == 'computer':
#             facility_name = 'Central Computer Facility'
#         elif facility_name == 'seminar':
#             facility_name = 'Seminar Hall'

#         desc = response.css('p.px-2.w-full::text').get()
#         print(desc)
#         self.facilities_data[f"Information about the {facility_name}"] = desc
#         self.total_fac["information About the facilties at College of Engineering Poonjar"] = self.facilities_data


#     def closed(self, response):
#         with open('college_json_data/cep.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_fac)
#         with open('college_json_data/cep.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEP_Placement(scrapy.Spider):
#     name = 'Placement'
#     start_urls = [
#         'https://cep.ac.in/placement',
#     ]

#     facilities_data = {}
#     total_placement_data = {}

#     def parse(self, response):
#         placement_data = {}
#         desc = response.css('div.flex.flex-col p.px-2.w-full::text').getall()
#         placement_data["Information or description about the Career Guidance and Placement Unit"] = ' '.join(desc)
#         faculty_table = response.css('table tbody tr td::text').getall()
#         print(faculty_table)
#         faculty_data = []
#         no = 1
#         for i in range(0,len(faculty_table),3):
#             faculty_data.append(
#                 {
#                     f"Faculty Name ({no})": faculty_table[i],
#                     f"Designation ({no})": faculty_table[i+1],
#                     f"Email({no})": faculty_table[i+2],
#                 }
#             )
#             no+=1
#         placement_data["Information about the faculty related with the Career Guidance and Placement Unit"] = faculty_data
#         self.total_placement_data = placement_data

#     def closed(self, response):
#         with open('college_json_data/cep.json', 'r') as f:
#             data = json.load(f)
#             data.append(self.total_placement_data)
#         with open('college_json_data/cep.json', 'w') as f:
#             json.dump(data,f,indent=4)

# class CEP_latest_news(scrapy.Spider):
#     name = 'Latest News'
#     start_urls = [
#         'https://cep.ac.in',
#     ]

#     latest_news = {}

#     def parse(self, response):
#         news = response.xpath("//p[contains(@class, 'text-balance py-[10px] pl-[10px]')]/text()").getall()
#         print(news)
        
#         self.latest_news['Latest News about College of Engineering Poonjar'] = news

#     def closed(self, response):
#         with open('college_json_data/cep.json', 'r') as f:
#             data = json.load(f)
#             data.update(self.latest_news)
#         with open('college_json_data/cep.json', 'w') as f:
#             json.dump(data,f,indent=4)


# # text-balance py-[10px] pl-[10px]

import scrapy
import json
import os
import re
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

def clean_text(item):
    item = re.sub(r'\\u[0-9a-fA-F]{4}', '', item)
    item = re.sub(r'[\u2013\u2019\u2022\u201a\u201c\u201d\u00b1\xa0]', '-', item)
    item = re.sub(r'\s+', ' ', item).strip()
    item = re.sub(r'<br>','',item)
    item = re.sub(r'-','',item).strip()
    return item

class aboutSpider(scrapy.Spider):
    name = "about"
    start_urls = ["https://cep.ac.in/about/aboutUs"]

    def parse(self, response):
        college_data = {
            # "title": response.css("h1::text").get(),
            "introduction of College of Engineering Poonjar": response.css("p.w-full::text").get(),
            # "vision_title": response.css("h1:contains('Vision')::text").get(),
            # "vision description of College of Engineering Poonjar": response.css("div:contains('Scientific advancement')::text").get(),
            # "mission_title": response.css("h1:contains('Mission')::text").get(),
            # "mission_point_1 of College of Engineering Poonjar": response.css("div ul li:nth-child(1)::text").get(),
            # "mission_point_2 of College of Engineering Poonjar": response.css("div ul li:nth-child(2)::text").get(),
            # "mission_point_3 of College of Engineering Poonjar": response.css("div ul li:nth-child(3)::text").get(),
            # "document_text": response.css("a[target='_blank'] div p::text").get(),
            # "document_link": response.urljoin(response.css("a[target='_blank']::attr(href)").get()),
        }

        file_path = "college_json_data/cep.json"

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            data = []
        data.update(college_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        self.log(f"Data has been appended to {file_path}")

class PrincipalDetailsSpider(scrapy.Spider):
    name = "principal_details"
    start_urls = ["https://cep.ac.in/about/principal"]

    def parse(self, response):
        principal_details = {
            "principal_name of College of Engineering Poonjar": response.css("span[class*='text-']::text").get(),
            # "designation": response.css("span.block.font-medium::text").get(),
            # "image_url of principal of College of Engineering Poonjar": response.css("img[class*='w-']::attr(src)").get(),
            # "message from principal of College of Engineering Poonjar": response.css("p.text-justify::text").get(),
            "email of principal of College of Engineering Poonjar": response.css("td:contains('Email :') + td::text").get(),
            "phone of principal of College of Engineering Poonjar": response.css("td:contains('Phone :') + td::text").get(),
        }

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}


        else:
            existing_data = {}
        existing_data.update(principal_details)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Principal details have been saved to {file_path}")

class ContactDetailsSpider(scrapy.Spider):
    name = "contact_details"
    start_urls = ["https://cep.ac.in/about/contact"]

    def parse(self, response):
        address = [line.get() for line in response.css("h3:contains('Address') + p::text, h3:contains('Address') + p ~ p::text")[:4]]
        phone_numbers = [phone.get() for phone in response.css("h3:contains('Phone') + p::text, h3:contains('Phone') + p + p::text")]
        email_addresses = [email.get() for email in response.css("h3:contains('Email') + p::text, h3:contains('Email') + p ~ p::text")]
        contact_details = {
            # "principal_name of College of Engineering Poonjar": response.css("h3:contains('Principal') + p::text").get(),
            # "principal_phone of College of Engineering Poonjar": response.css("h3:contains('Principal') + p + p::text").get().replace("Phone: ", ""),
            
            "placement_officer_name of College of Engineering Poonjar": response.css("h3:contains('Placement Officer') + p::text").get(),
            "placement_officer_phone of College of Engineering Poonjar": response.css("h3:contains('Placement Officer') + p + p::text").get().replace("Phone: ", ""),
            "placement_officer_email of College of Engineering Poonjar": response.css("h3:contains('Placement Officer') + p + p + p::text").get().replace("Email: ", ""),
            
            # "address of College of Engineering Poonjar": [
            #     line.get() for line in response.css("h3:contains('Address') + p::text, h3:contains('Address') + p ~ p::text")[:4]
            # ],
            
            # "phone_numbers of College of Engineering Poonjar": [
            #     phone.get() for phone in response.css("h3:contains('Phone') + p::text, h3:contains('Phone') + p + p::text")
            # ],
            
            # "email_addresses of College of Engineering Poonjar": [
            #     email.get() for email in response.css("h3:contains('Email') + p::text, h3:contains('Email') + p ~ p::text")
            # ]
            "address of College of Engineering Poonjar": " ".join(address),
            "phone_numbers of College of Engineering Poonjar": ", ".join(phone_numbers),
            "email_addresses of College of Engineering Poonjar": ", ".join(email_addresses)
        }

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}
            
        existing_data.update(contact_details)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Contact details have been saved to {file_path}")

class ComputerScienceDepartmentSpider(scrapy.Spider):
    name = "computer_science_department"
    start_urls = ["https://cep.ac.in/departments/cs"]

    def parse(self, response):
        # department_name = response.css("h1.text-center::text").get().strip()
        department_introduction = response.css("div.text-justify::text").get().strip()
    #     department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


    #     department_mission = [
    #     mission.strip() for mission in response.css("div.grid div:nth-child(2) ul li::text").getall()
    # ]


        faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
        faculty_flat = [
            {
                "faculty_name": faculty.css("span.text-dark::text").get().strip(),
                "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
                "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
                # "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
                "faculty_profile_link": faculty.css("a::attr(href)").get()
            }
            for faculty in faculty_details
        ]
        lab_staff_details = faculty_details[-5:]
        lab_staff_flat = [
            {
                "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
                "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
                "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
                # "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
            }
            for staff in lab_staff_details
        ]

        flattened_data = {
            # "department_name of Computer Science department": department_name,
            "department_introduction of Computer Science department": department_introduction,
            # "department_vision of Computer Science department": department_vision,
            # "department_mission of Computer Science department": "; ".join(department_mission), 
        }

        for i, faculty in enumerate(faculty_flat, start=1):
            flattened_data.update({
                f"faculty_{i}_name of Computer Science department": faculty["faculty_name"],
                f"faculty_{i}_designation of Computer Science department": faculty["faculty_designation"],
                f"faculty_{i}_qualification of Computer Science department": faculty["faculty_qualification"],
                # f"faculty_{i}_image_url of Computer Science department": faculty["faculty_image_url"],
                f"faculty_{i}_profile_link of Computer Science department": faculty["faculty_profile_link"]
            })

        # Add lab staff details to flattened data
        for i, staff in enumerate(lab_staff_flat, start=1):
            flattened_data.update({
                f"lab_staff_{i}_name of Computer Science department": staff["lab_staff_name"],
                f"lab_staff_{i}_designation of Computer Science department": staff["lab_staff_designation"],
                f"lab_staff_{i}_qualification of Computer Science department": staff["lab_staff_qualification"],
                # f"lab_staff_{i}_image_url of Computer Science department": staff["lab_staff_image_url"]
            })

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}
        existing_data.update(flattened_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Computer Science department details have been saved to {file_path}")

class ELectricalElectronicsDepartmentSpider(scrapy.Spider):
    name = "electrical_electronics_department"
    start_urls = ["https://cep.ac.in/departments/ee"]

    def parse(self, response):
        # department_name = response.css("h1.text-center::text").get().strip()
        department_introduction = response.css("div.text-justify::text").get().strip()
    #     department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


    #     department_mission = [
    #     mission.strip() for mission in response.css("div img[src*='mission.png'] + h1 + div ul.list-disc li::text").getall()
    # ]


        faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
        faculty_flat = [
            {
                "faculty_name": faculty.css("span.text-dark::text").get().strip(),
                "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
                "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
                # "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
                "faculty_profile_link": faculty.css("a::attr(href)").get()
            }
            for faculty in faculty_details
        ]
        lab_staff_details = faculty_details[-5:]
        lab_staff_flat = [
            {
                "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
                "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
                "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
                # "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
            }
            for staff in lab_staff_details
        ]

        flattened_data = {
            # "department_name of ELectrical and ELectronics department": department_name,
            "department_introduction of ELectrical and ELectronics department": department_introduction,
            # "department_vision of ELectrical and ELectronics department": department_vision,
            # "department_mission of ELectrical and ELectronics department": "; ".join(department_mission), 
        }

        for i, faculty in enumerate(faculty_flat, start=1):
            flattened_data.update({
                f"faculty_{i}_name of ELectrical and ELectronics department": faculty["faculty_name"],
                f"faculty_{i}_designation of ELectrical and ELectronics department": faculty["faculty_designation"],
                f"faculty_{i}_qualification of ELectrical and ELectronics department": faculty["faculty_qualification"],
                # f"faculty_{i}_image_url of ELectrical and ELectronics department": faculty["faculty_image_url"],
                f"faculty_{i}_profile_link of ELectrical and ELectronics department": faculty["faculty_profile_link"]
            })

        # Add lab staff details to flattened data
        for i, staff in enumerate(lab_staff_flat, start=1):
            flattened_data.update({
                f"lab_staff_{i}_name of ELectrical and ELectronics department": staff["lab_staff_name"],
                f"lab_staff_{i}_designation of ELectrical and ELectronics department": staff["lab_staff_designation"],
                f"lab_staff_{i}_qualification of ELectrical and ELectronics department": staff["lab_staff_qualification"],
                # f"lab_staff_{i}_image_url of ELectrical and ELectronics department": staff["lab_staff_image_url"]
            })

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}
        existing_data.update(flattened_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"ELectrical and ELectronics department details have been saved to {file_path}")


class Electronics_CommunicationDepartmentSpider(scrapy.Spider):
    name = "electronics_communication_department"
    start_urls = ["https://cep.ac.in/departments/ec"]

    def parse(self, response):
        # department_name = response.css("h1.text-center::text").get().strip()
        department_introduction = response.css("div.text-justify::text").get().strip()
    #     department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


    #     department_mission = [
    #     mission.strip() for mission in response.css("div img[src*='mission.png'] + h1 + div ul.list-disc li::text").getall()
    # ]


        faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
        faculty_flat = [
            {
                "faculty_name": faculty.css("span.text-dark::text").get().strip(),
                "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
                "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
                # "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
                "faculty_profile_link": faculty.css("a::attr(href)").get()
            }
            for faculty in faculty_details
        ]
        lab_staff_details = faculty_details[-5:]
        lab_staff_flat = [
            {
                "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
                "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
                "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
                # "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
            }
            for staff in lab_staff_details
        ]

        flattened_data = {
            # "department_name of ELectrical and ELectronics department": department_name,
            "department_introduction of Electronics and communication department": department_introduction,
            # "department_vision of Electronics and communication department": department_vision,
            # "department_mission of Electronics and communication department": "; ".join(department_mission), 
        }

        for i, faculty in enumerate(faculty_flat, start=1):
            flattened_data.update({
                f"faculty_{i}_name of Electronics and communication department": faculty["faculty_name"],
                f"faculty_{i}_designation of Electronics and communication department": faculty["faculty_designation"],
                f"faculty_{i}_qualification of Electronics and communication department": faculty["faculty_qualification"],
                # f"faculty_{i}_image_url of Electronics and communication department": faculty["faculty_image_url"],
                f"faculty_{i}_profile_link of Electronics and communication department": faculty["faculty_profile_link"]
            })

        # Add lab staff details to flattened data
        for i, staff in enumerate(lab_staff_flat, start=1):
            flattened_data.update({
                f"lab_staff_{i}_name of Electronics and communication department": staff["lab_staff_name"],
                f"lab_staff_{i}_designation of Electronics and communication department": staff["lab_staff_designation"],
                f"lab_staff_{i}_qualification of Electronics and communication department": staff["lab_staff_qualification"],
                # f"lab_staff_{i}_image_url of Electronics and communication department": staff["lab_staff_image_url"]
            })

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}
        existing_data.update(flattened_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Electronics and communication department details have been saved to {file_path}")


class AppliedScienceDepartmentSpider(scrapy.Spider):
    name = "applied_science_department"
    start_urls = ["https://cep.ac.in/departments/sah"]

    def parse(self, response):
        # department_name = response.css("h1.text-center::text").get().strip()
        department_introduction = response.css("div.text-justify::text").get().strip()
    #     department_vision = response.xpath("//*[contains(text(), 'Vision')]/ancestor::div[contains(@class, 'grid')]/div[1]/div[2]/text()").getall()


    #     department_mission = [
    #     mission.strip() for mission in response.css("div img[src*='mission.png'] + h1 + div ul.list-disc li::text").getall()
    # ]


        faculty_details = response.css("div.flex.flex-col.mr-5.text-center.mb-11")
        faculty_flat = [
            {
                "faculty_name": faculty.css("span.text-dark::text").get().strip(),
                "faculty_designation": faculty.css("span.block.font-medium::text").get().strip(),
                "faculty_qualification": faculty.css("span.block.text-sm::text").get().strip(),
                # "faculty_image_url": faculty.css("img::attr(src)").get().strip(),
                "faculty_profile_link": faculty.css("a::attr(href)").get()
            }
            for faculty in faculty_details
        ]
        lab_staff_details = faculty_details[-5:]
        lab_staff_flat = [
            {
                "lab_staff_name": staff.css("span.text-dark::text").get().strip(),
                "lab_staff_designation": staff.css("span.block.font-medium::text").get().strip(),
                "lab_staff_qualification": staff.css("span.block.text-sm::text").get().strip(),
                # "lab_staff_image_url": staff.css("img::attr(src)").get().strip()
            }
            for staff in lab_staff_details
        ]

        flattened_data = {
            # "department_name of ELectrical and ELectronics department": department_name,
            "department_introduction of Applied Science department": department_introduction,
            # "department_vision of Applied Science department": department_vision,
            # "department_mission of Applied Science department": "; ".join(department_mission), 
        }

        for i, faculty in enumerate(faculty_flat, start=1):
            flattened_data.update({
                f"faculty_{i}_name of Applied Science department": faculty["faculty_name"],
                f"faculty_{i}_designation of Applied Science department": faculty["faculty_designation"],
                f"faculty_{i}_qualification of Applied Science department": faculty["faculty_qualification"],
                # f"faculty_{i}_image_url of Applied Science department": faculty["faculty_image_url"],
                f"faculty_{i}_profile_link of Applied Science department": faculty["faculty_profile_link"]
            })

        # Add lab staff details to flattened data
        for i, staff in enumerate(lab_staff_flat, start=1):
            flattened_data.update({
                f"lab_staff_{i}_name of Applied Science department": staff["lab_staff_name"],
                f"lab_staff_{i}_designation of Applied Science department": staff["lab_staff_designation"],
                f"lab_staff_{i}_qualification of Applied Science department": staff["lab_staff_qualification"],
                # f"lab_staff_{i}_image_url of Applied Science department": staff["lab_staff_image_url"]
            })

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}
        existing_data.update(flattened_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Applied Science department details have been saved to {file_path}")        


class IQACSpider(scrapy.Spider):
    name = "iqac_spider"
    start_urls = ["https://cep.ac.in/iqac"] 

    def parse(self, response):
        about_iqac = response.xpath("//h2[contains(text(), 'About IQAC')]/following-sibling::p/text()").get()

        members = response.xpath("//table[contains(@class, 'min-w-full')]/tbody/tr")
        member_data = []
        
        # Collect title and name pairs
        for i in range(0, len(members), 2):
            title = members[i].xpath("./td[@colspan='2']/text()").get()
            name = members[i+1].xpath("./td[@class='px-6 py-4 text-sm font-medium text-gray-800 dark:text-gray-200 text-center']/text()").get()

            if title and name:
                member_data.append({
                    "title": title.strip() if title else None,
                    "name": name.strip() if name else None
                })

        flattened_data = {
            "about_iqac": about_iqac.strip() if about_iqac else None,
        }
        # for i, member in enumerate(member_data, start=1):
        #     flattened_data.update({
        #         f"IQAC {member['title']} member_{i}_name": member["name"]
        #     })

        file_path = "college_json_data/cep.json"

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}

        existing_data.update(flattened_data)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"IQAC department details have been saved to {file_path}")

class AcademicCouncilSpider(scrapy.Spider):
    name = "academic_council"
    start_urls = ["https://cep.ac.in/council"] 

    # def parse(self, response):

    #     academic_council_intro = response.xpath("//div[@class='max-w-[1200px] mx-auto']/p/text()").get()

    #     functions = response.xpath("//h2[contains(text(), 'The functions of the Academic Council are as below:')]/following-sibling::ul[1]/li/text()").getall()
    #     flattened_data = {
    #         "academic_council_intro": academic_council_intro.strip() if academic_council_intro else None,
    #         "academic_council_functions": functions if functions else [],
    #     }
    def parse(self, response):
        academic_council_intro = response.xpath("//div[@class='max-w-[1200px] mx-auto']/p/text()").get()

        functions = response.xpath("//h2[contains(text(), 'The functions of the Academic Council are as below:')]/following-sibling::ul[1]/li/text()").getall()

        flattened_functions = {
            f"function_{i+1}_of_academic_council": function.strip()
            for i, function in enumerate(functions)
        }

        flattened_data = {
            "academic_council_intro": academic_council_intro.strip() if academic_council_intro else None,
        }

        flattened_data.update(flattened_functions)
        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}

        existing_data.update(flattened_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Academic Council details have been saved to {file_path}")        


class ProgramsOfferedSpider(scrapy.Spider):
    name = "programs_offered"
    start_urls = ["https://cep.ac.in/programs"]

    def parse(self, response):

        post_graduate_courses = response.xpath("//h2[contains(text(), 'Post Graduate')]/following-sibling::div//table/tbody/tr")
        post_graduate_data = {}
        for i, course in enumerate(post_graduate_courses):
            course_name = course.xpath(".//td[1]/text()").get().strip()
            intake_number = course.xpath(".//td[2]/text()").get().strip()
            # post_graduate_data[f"pg_course_{i+1}_name"] = course_name
            post_graduate_data[f"pg_course_{i+1} {course_name}_intake,seat matrix"] = intake_number

        under_graduate_courses = response.xpath("//h2[contains(text(), 'Under Graduate')]/following-sibling::div//table/tbody/tr")
        under_graduate_data = {}
        for i, course in enumerate(under_graduate_courses):
            course_name = course.xpath(".//td[1]/text()").get().strip()
            intake_number = course.xpath(".//td[2]/text()").get().strip()
            # under_graduate_data[f"ug_course_{i+1}_name"] = course_name
            under_graduate_data[f"ug_course_{i+1} {course_name}_intake,seat matrix"] = intake_number

        diploma_courses = response.xpath("//h2[contains(text(), 'Diploma')]/following-sibling::div//table/tbody/tr")
        diploma_data = {}
        for i, course in enumerate(diploma_courses):
            course_name = course.xpath(".//td[1]/text()").get().strip()
            intake_number = course.xpath(".//td[2]/text()").get().strip()
            # diploma_data[f"diploma_course_{i+1}_name"] = course_name
            diploma_data[f"diploma_course_{i+1} {course_name}_intake,seat matrix"] = intake_number

        flattened_data = {
            **post_graduate_data,
            **under_graduate_data,
            **diploma_data
        }

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}

        existing_data.update(flattened_data)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Programs offered details have been saved to {file_path}")

class AdmissionDetailsSpider(scrapy.Spider):
    name = "admission_details"
    start_urls = ["https://cep.ac.in/admission"]  

    def parse(self, response):
        btech_admission_data = {
            "btech_academic_eligibility": response.xpath("//h2[contains(text(), 'B.Tech Admission')]/following-sibling::p[1]//text()").get().strip(),
            "btech_allotment": response.xpath("//h2[contains(text(), 'B.Tech Admission')]/following-sibling::p[2]//text()").get().strip(),
            "btech_exam_link": response.xpath("//h2[contains(text(), 'B.Tech Admission')]/following-sibling::div[1]//a/@href").get(),
        }

        diploma_admission_data = {
            "diploma_academic_eligibility": response.xpath("//h2[contains(text(), 'Diploma Admission')]/following-sibling::p[1]//text()").get().strip(),
            "diploma_allotment": response.xpath("//h2[contains(text(), 'Diploma Admission')]/following-sibling::p[2]//text()").get().strip(),
            "diploma_exam_link": response.xpath("//h2[contains(text(), 'Diploma Admission')]/following-sibling::div[1]//a/@href").get(),
        }

        mca_admission_data = {
            "mca_academic_eligibility": response.xpath("//h2[contains(text(), 'MCA Admission')]/following-sibling::p[1]//text()").get().strip(),
            "mca_allotment": response.xpath("//h2[contains(text(), 'MCA Admission')]/following-sibling::p[2]//text()").get().strip(),
            "mca_exam_link": response.xpath("//h2[contains(text(), 'MCA Admission')]/following-sibling::div[1]//a/@href").get(),
        }

        lateral_entry_admission_data = {
            "lateral_entry_academic_eligibility": response.xpath("//h2[contains(text(), 'B.Tech Lateral Entry Admission')]/following-sibling::p[1]//text()").get().strip(),
            "lateral_entry_allotment": response.xpath("//h2[contains(text(), 'B.Tech Lateral Entry Admission')]/following-sibling::p[2]//text()").get().strip(),
            "lateral_entry_exam_link": response.xpath("//h2[contains(text(), 'B.Tech Lateral Entry Admission')]/following-sibling::div[1]//a/@href").get(),
        }

        contact_data = {}
        contact_names = [
            response.xpath("//table/tbody/tr[1]/td[1]//text()").get().strip(),
            response.xpath("//table/tbody/tr[2]/td[1]//text()").get().strip(),
            response.xpath("//table/tbody/tr[3]/td[1]//text()").get().strip(),
        ]
        contact_numbers = [
            response.xpath("//table/tbody/tr[1]/td[2]//text()").get().strip(),
            response.xpath("//table/tbody/tr[2]/td[2]//text()").get().strip(),
            response.xpath("//table/tbody/tr[3]/td[2]//text()").get().strip(),
        ]

        for i, (name, number) in enumerate(zip(contact_names, contact_numbers), start=1):
            contact_data[f"contact_{i}_name"] = name
            contact_data[f"{name}_number"] = number

        fee_structure_data = {
            "mca_fee_link": response.xpath("//table[contains(@class, 'min-w-full')]//tr[1]/td[2]//a/@href").get(),
            "btech_fee_link": response.xpath("//table[contains(@class, 'min-w-full')]//tr[2]/td[2]//a/@href").get(),
            "diploma_fee_link": response.xpath("//table[contains(@class, 'min-w-full')]//tr[3]/td[2]//a/@href").get(),
        }

        flattened_data = {
            **btech_admission_data,
            **diploma_admission_data,
            **mca_admission_data,
            **lateral_entry_admission_data,
            **contact_data,
            **fee_structure_data,
        }

        file_path = "college_json_data/cep.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, dict):
                        self.log("Existing file is not a dictionary. Replacing content.")
                        existing_data = {}
                except json.JSONDecodeError:
                    self.log("JSON file is corrupted or empty. Initializing as an empty dictionary.")
                    existing_data = {}
        else:
            existing_data = {}

        existing_data.update(flattened_data)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        self.log(f"Admission details have been saved to {file_path}")             

class CEP_facilities(scrapy.Spider):
    name = 'facilities'
    start_urls = [
        'https://cep.ac.in/facilities/computer',
        'https://cep.ac.in/facilities/library',
        'https://cep.ac.in/facilities/seminar',
        'https://cep.ac.in/facilities/transport',
        'https://cep.ac.in/facilities/hostel',
        'https://cep.ac.in/facilities/canteen'
    ]

    facilities_data = {}
    total_fac = {}

    def parse(self, response):
        facility_name = response.url.split('/')[-1]
        if facility_name == 'computer':
            facility_name = 'Central Computer Facility'
        elif facility_name == 'seminar':
            facility_name = 'Seminar Hall'

        desc = response.css('p.px-2.w-full::text').get()
        self.facilities_data[f"Information about the {facility_name}"] = desc
        self.total_fac["information About the facilties at College of Engineering Poonjar"] = self.facilities_data


    def closed(self, response):
        with open('college_json_data/cep.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_fac)
        with open('college_json_data/cep.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEP_Placement(scrapy.Spider):
    name = 'Placement'
    start_urls = [
        'https://cep.ac.in/placement',
    ]

    facilities_data = {}
    total_placement_data = {}

    def parse(self, response):
        placement_data = {}
        desc = response.css('div.flex.flex-col p.px-2.w-full::text').getall()
        placement_data["Information or description about the Career Guidance and Placement Unit"] = ' '.join(desc)
        faculty_table = response.css('table tbody tr td::text').getall()
        print(faculty_table)
        faculty_data = []
        no = 1
        for i in range(0,len(faculty_table),3):
            faculty_data.append(
                {
                    f"Faculty Name ({no})": faculty_table[i],
                    f"Designation ({no})": faculty_table[i+1],
                    f"Email({no})": faculty_table[i+2],
                }
            )
            no+=1
        placement_data["Information about the faculty related with the Career Guidance and Placement Unit"] = faculty_data
        self.total_placement_data = placement_data

    def closed(self, response):
        with open('college_json_data/cep.json', 'r') as f:
            data = json.load(f)
            data.append(self.total_placement_data)
        with open('college_json_data/cep.json', 'w') as f:
            json.dump(data,f,indent=4)

class CEP_latest_news(scrapy.Spider):
    name = 'Latest News'
    start_urls = [
        'https://cep.ac.in',
    ]

    latest_news = {}

    def parse(self, response):
        news = response.xpath("//p[contains(@class, 'text-balance py-[10px] pl-[10px]')]/text()").getall()
        print(news)
        
        self.latest_news['Latest News about College of Engineering Poonjar'] = news

    def closed(self, response):
        with open('college_json_data/cep.json', 'r') as f:
            data = json.load(f)
            data.update(self.latest_news)
        with open('college_json_data/cep.json', 'w') as f:
            json.dump(data,f,indent=4)


