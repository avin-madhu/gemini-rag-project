import scrapy
import json
from urllib.parse import urljoin
import scrapy.spiderloader
import os

def clear_file():
    with open('college_json_data/cec.json','w') as f:
        f.write('[]')
    with open('college_json_data/cek.json','w') as f:
        f.write('[]')
    with open('college_json_data/mec.json','w') as f:
        f.write('[]')
    with open('college_json_data/casa.json','w') as f:
        f.write('[]')
    with open('college_json_data/cekp.json','w') as f:
        f.write('[]')
    with open('college_json_data/cep.json','w') as f:
        f.write('[]')
    with open('college_json_data/cech.json','w') as f:
        f.write('[]')

# spider to get the details from the admission section
class CEC_AdmissionsSpider(scrapy.Spider):
    name = 'admissions'
    allowed_domains = ['ceconline.edu']
    start_urls = ['https://ceconline.edu/admission/btech_admissions/']

    # This will hold all flat admission data
    total_admission_data = {}

    def parse(self, response):
        flat_data = {}

        # Extract common elements from the page
        admission_document_list_link = response.css('div.elementor-widget-container p a::attr(href)').get()
        page_header = response.css('div.wpb_wrapper h2::text').get()
        page_admission_description = response.css('div.wpb_wrapper p::text').get()
        seat_capacity_table = response.css('tr td span::text').getall()
        btech_eligibility_criteria = response.css('ul li span::text').getall()

        # Flatten the department seat capacity data.
        for i in range(0, len(seat_capacity_table), 2):
            department = seat_capacity_table[i].strip()
            capacity = seat_capacity_table[i + 1].strip().replace('\u00a0', '')
            flat_data[f"seat_capacity_{department}"] = capacity

        # Combine eligibility criteria list into a single string
        eligibility_text = "".join(item.strip() for item in btech_eligibility_criteria).replace('\u00a0', '')
        flat_data['admission_description'] = page_admission_description.strip() if page_admission_description else ""
        flat_data['btech_eligibility_criteria'] = eligibility_text
        flat_data['documents_to_be_submitted_during_admission_link'] = (
            admission_document_list_link.replace(' ', '%20') if admission_document_list_link else ""
        )

        # Store the flattened data
        self.total_admission_data.update(flat_data)

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_admission_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_admission_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)    

class CEC_DepartmentSpider(scrapy.Spider):
    name = 'department'
    allowed_domains = ['ceconline.edu']
    start_urls = ['https://ceconline.edu/departments'] 

    total_department_data = {}  # Changed to flat dictionary

    def parse(self, response):
        department_links = response.css('figure.wpb_wrapper.vc_figure a::attr(href)').getall()
        for department_link in department_links:
            yield response.follow(department_link, self.parse_department)  

    def parse_department(self, response):
        # Get department basic info
        department_title = response.css('h2.vc_custom_heading.vc_do_custom_heading::text').get()
        if not department_title:
            return
        
        dept_key = department_title.lower().replace(' ', '_')
        
        # Flatten department description
        department_descriptions = response.css('div.wpb_text_column.wpb_content_element p::text').getall()
        self.total_department_data[f"{dept_key}_description"] = " ".join(department_descriptions).strip()

        # Flatten HOD details
        # self.total_department_data[f"{dept_key}_hod_image_url"] = response.css('figure.wpb_wrapper.vc_figure div img::attr(src)').get() or ""
        self.total_department_data[f"{dept_key}_hod_name"] = response.css('div.stm-teacher-bio__text p span strong::text').get() or ""
        self.total_department_data[f"{dept_key}_hod_designation"] = response.css('div.stm-teacher-bio__text p strong span::text').get() or ""
        self.total_department_data[f"{dept_key}_hod_email"] = response.css('li.stm-contact-details__item.stm-contact-details__item_type_email a::text').get() or ""

        # Flatten faculty data
        # CS Faculty
        faculty_list = response.css('td[width="319"] a::text,td[width="208"]::text,td[width="97"]::text').getall()
        if faculty_list:
            i = 2
            faculty_count = 1
            while i < len(faculty_list) and int(faculty_list[i]) < 30:
                faculty_id = faculty_list[i].strip()
                faculty_name = faculty_list[i+1].replace('\u00a0', '').strip()
                faculty_designation = faculty_list[i+2].strip()
                
                # self.total_department_data[f"{dept_key}_faculty_{faculty_count}_id"] = faculty_id
                self.total_department_data[f"{dept_key}_faculty_{faculty_count}_name"] = faculty_name
                self.total_department_data[f"{dept_key}_faculty_{faculty_count}_designation"] = faculty_designation
                
                faculty_count += 1
                i += 3

        # EC Faculty
        faculty_list = response.css('table[width="685"] td::text, table[width="685"] td a::text').getall()
        if faculty_list:
            faculty_count = 1
            i = 0
            while i < len(faculty_list) - 2:
                faculty_id = faculty_list[i].strip()
                faculty_name = faculty_list[i+1].strip()
                faculty_designation = faculty_list[i+2].replace('\u00a0', '').strip()
                
                # self.total_department_data[f"{dept_key}_ec_faculty_{faculty_count}_id"] = faculty_id
                self.total_department_data[f"{dept_key}_ec_faculty_{faculty_count}_name"] = faculty_name
                self.total_department_data[f"{dept_key}_ec_faculty_{faculty_count}_designation"] = faculty_designation
                
                faculty_count += 1
                i += 3

        # EEE Faculty
        faculty_list = response.css('table.alt td a::text, table.alt td::text').getall()[1:40]
        if faculty_list and faculty_list[0][0] not in "4D":
            faculty_count = 1
            i = 0
            while i < len(faculty_list) - 2:
                faculty_id = faculty_list[i].strip()
                faculty_name = faculty_list[i+1].strip()
                faculty_designation = faculty_list[i+2].replace('\u00a0', '').strip()
                
                # self.total_department_data[f"{dept_key}_eee_faculty_{faculty_count}_id"] = faculty_id
                self.total_department_data[f"{dept_key}_eee_faculty_{faculty_count}_name"] = faculty_name
                self.total_department_data[f"{dept_key}_eee_faculty_{faculty_count}_designation"] = faculty_designation
                
                faculty_count += 1
                i += 3

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_department_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_department_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_placementSpider(scrapy.Spider):
    name = "placementSpider"
    allowed_domains = ['ceconline.edu']
    start_urls = ['https://ceconline.edu/placement']

    total_placement_data = {}

    def parse(self, response):
        placement_cell_description = response.css('div.wpb_column.vc_column_container.vc_col-sm-12 p::text').getall()
        placement_cell_activities = response.css('div.wpb_column.vc_column_container.vc_col-sm-12 li::text').getall()

        # Flatten into simple key-value pairs
        self.total_placement_data["placement_cell_description"] = ' '.join(placement_cell_description[:-2]).strip()
        self.total_placement_data["placement_cell_activities"] = ' '.join(placement_cell_activities[:-5]).strip()

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_placement_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_placement_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_organisationSpider(scrapy.Spider):
    name = "organisationSpider"
    allowed_domains = ['ceconline.edu']
    start_urls = ['https://ceconline.edu/organizations']

    total_organisation_data = {}
    org_count = 0  # Counter for organizations

    def parse(self, response):
        organisation_links = response.css('div.wpb_wrapper h3 a::attr(href)').getall()
        organisation_links = [i for i in organisation_links if i]  # Remove empty strings
        for organisation_link in organisation_links:
            yield response.follow(organisation_link, self.parse_organisation)  

    def parse_organisation(self, response):
        # Get organization title and normalize it
        organisation_title = response.css('h1.vc_custom_heading.vc_do_custom_heading::text, h2.vc_custom_heading.vc_do_custom_heading::text, div.stm-title.stm-title_sep_bottom::text, h2.vc_custom_heading.vc_do_custom_heading a::text').get()
        if "Alumni" in organisation_title:
            organisation_title = "Alumni"
        
        org_key = organisation_title.lower().replace(' ', '_').replace('-', '_')
        
        # Get website link
        organisation_website_link = response.css('div.wpb_wrapper p a::attr(href)').get(default="No Link")

        # Get and clean description
        organisation_description = response.css('div.wpb_text_column.wpb_content_element p::text, div.wpb_text_column.wpb_content_element p span::text').getall()
        
        # Clean the description
        unwanted_chars = ["\u201c", "\u2019", "\u2013", "\u00a0", "\u201d", "\u2018", "\u201d"]
        cleaned_description = []
        
        for part in organisation_description:
            if isinstance(part, str):
                for char in unwanted_chars:
                    part = part.replace(char, " ")
                if part.strip() and part.strip() != '.':
                    cleaned_description.append(part.strip())

        # Store data in flat structure
        self.org_count += 1
        # self.total_organisation_data[f"organization_{self.org_count}_name"] = organisation_title
        self.total_organisation_data[f"organization_{self.org_count}_{organisation_title}_description"] = ' '.join(cleaned_description).strip()
        self.total_organisation_data[f"organization_{self.org_count}_{organisation_title}_website"] = organisation_website_link

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_organisation_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_organisation_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_principal_spider(scrapy.Spider):
    name = 'principal_spider'
    allowed_domains = ['ceconline.edu']
    start_urls = ['https://ceconline.edu/administrators/hari/']

    total_principal_details = {}

    def parse(self, response):
        # Basic info
        self.total_principal_details['principal_name'] = response.css('.stm-title.stm-title_sep_bottom::text').extract_first().strip()

        # Flatten qualifications
        qualification_titles = response.css('.wpb_wrapper ul span::text').getall()
        qualification_details = [q.strip() for q in response.css('.wpb_wrapper ul li::text').getall()]
        for i, (title, detail) in enumerate(zip(qualification_titles, qualification_details), 1):
            # self.total_principal_details[f'principal_qualification_{i}_title'] = title.strip()
            self.total_principal_details[f'principal_qualification_{i}_{title.strip()}_detail'] = detail.strip()
            if i==1:
                break

        # Flatten contact details
        contact_details = response.css('.stm-contact-details__items li')
        for detail in contact_details:
            text = detail.css('::text').get().strip()
            key = detail.css('::attr(class)').get().split('_')[-1]
            if text:
                self.total_principal_details[f'principal_contact_{key.replace("-", "_")}'] = text

        # Flatten positions
        # positions = response.css('.vc_tta-panel-body ol li::text').getall()
        # for i, position in enumerate(positions, 1):
        #     self.total_principal_details[f'principal_position_{i}'] = position.strip()

        # Flatten expertise fields
        fields = response.css('.vc_tta-panel-body:contains("LabVIEW") .wpb_wrapper ul li::text').getall()
        for i, field in enumerate(fields, 1):
            self.total_principal_details[f'principal_expertise_{i}'] = field.strip()

        # Flatten industry interactions
        # interactions = response.css('.vc_tta-panel-body:contains("Minimization of atomic norm in vibration signals") .wpb_wrapper ul li em::text').getall()
        # for i, interaction in enumerate(interactions, 1):
        #     self.total_principal_details[f'principal_industry_interaction_{i}'] = interaction.strip()

        # Flatten research activities
        research = response.css('.vc_tta-panel-body:contains("Incorporated a company Rand Walk Research") .wpb_wrapper ul li p::text').getall()
        for i, activity in enumerate(research, 1):
            self.total_principal_details[f'principal_research_activity_{i}'] = activity.strip()

        # Flatten books
        books = response.css('.vc_tta-panel-body:contains("Electronics Laboratory Handbook with Simulations using Quite Universal Circuit Simulator (Qucs), Authors Press,New Delhi,ISBN-978-93-5529-048-9") .wpb_wrapper ol li::text').getall()
        for i, book in enumerate(books, 1):
            self.total_principal_details[f'principal_book_{i}'] = book.strip()

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_principal_details, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_principal_details)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_PTASpider(scrapy.Spider):
    name = "pta_spider"
    start_urls = ["https://ceconline.edu/about/committees/pta/"]

    total_pta_data = {}

    def parse(self, response):
        pta_section = response.xpath('//div[contains(@class, "wpb_wrapper")]/div[contains(@class, "pta-section")]')

        if pta_section:
            tables = pta_section.xpath(".//table")

            for table_index, table in enumerate(tables):
                rows = table.xpath(".//tr[position()>1]")  # Skip header row
                member_count = 1
                
                for row in rows:
                    cols = row.xpath(".//td")
                    if cols:  # Check if the row has any columns at all
                        try:
                            name_parts = cols[1].xpath(".//text()").getall()
                            name = "".join(name_parts).strip()
                            phone = cols[2].xpath(".//text()").get("").strip()
                            # student_name = cols[3].xpath(".//text()").get("").strip() if len(cols) > 3 else ""

                            # Determine member type based on table index
                            member_type = "executive" if table_index == 0 else "faculty" if table_index == 1 else "special"
                            
                            # Store flattened data
                            self.total_pta_data[f'pta_{member_type}_member_{member_count}_name and phone number'] = (name,phone)
                            # self.total_pta_data[f'pta_{member_type}_member_{member_count}_phone'] = phone
                            # if student_name:
                            #     self.total_pta_data[f'pta_{member_type}_member_{member_count}_student'] = student_name
                            
                            member_count += 1
                            
                        except IndexError as e:
                            self.log(f"IndexError in table {table_index + 1}: {e}. Row content: {row.get()}")
                            continue
                    else:
                        self.log(f"Empty row found in table {table_index + 1}. Skipping.")

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_pta_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_pta_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_NoticeSpider(scrapy.Spider):
    name = 'notice_spider'
    start_urls = ['https://ceconline.edu/notifications-2/']

    total_notice_data = {}
    
    def parse(self, response):
        notice_identifiers = {
            'semester_registration_revised': 'Revised notice for  Semester Registration',
            'semester_registration': 'Semester Registration for B.Tech  and MCA',
            'fees_payment': 'Fees Payment for Semester registration',
            'induction_programme': 'Student Induction Programme 2024',
            'btech_documents': 'B.Tech Documents To Be Submitted',
            'ladies_hostel': 'Ladies Hostel List',
            'answer_book': 'Purchase Of Main Answer Book'
        }
        
        for notice_id, notice_text in notice_identifiers.items():
            notice_element = response.xpath(
                f"//div[contains(@class, 'elementor-widget-container')]"
                f"//h2[contains(@class, 'elementor-heading-title')]"
                f"//a[contains(., '{notice_text}')]"
            )
            
            if notice_element:
                link = notice_element.attrib.get('href', '')
                text = notice_element.css('u::text').get('').strip()
                
                if not text:
                    text = notice_element.get().strip()
                
                if link.startswith('/'):
                    link = urljoin(response.url, link)
                    
                doc_type = self.get_document_type(link)
                self.total_notice_data[f"notice_{notice_id}_title"] = text
                self.total_notice_data[f"notice_{notice_id}_url"] = link
                # self.total_notice_data[f"notice_{notice_id}_type"] = doc_type
                
                if doc_type == 'pdf':
                    yield scrapy.Request(
                        url=link,
                        callback=self.save_pdf,
                        meta={
                            'title': text,
                            'notice_id': notice_id
                        },
                        errback=self.handle_error
                    )

    def get_document_type(self, url):
        if url.endswith('.pdf'):
            return 'pdf'
        elif 'onlinesbi.sbi' in url:
            return 'payment_portal'
        elif url.startswith('http') or url.startswith('https'):
            return 'external_link'
        return 'other'

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_notice_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_notice_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_InternalQualityAssuranceSpider(scrapy.Spider):
    name = "iqac_spider"
    start_urls = ["https://ceconline.edu/about/committees/internal-quality-assurance-cell-or-internal-audit-cell-iqac-iac/#1608789190414-66d69914-5d08"]

    total_internal_quality_assurance_data = {}

    def parse(self, response):
        iqac_div = response.xpath(
            '//div[contains(@class, "wpb_text_column") and contains(@class, "wpb_content_element")]'
            '/div[contains(@class, "wpb_wrapper")]'
            '/h2/strong[contains(text(), "Internal Quality Assurance Cell")]/ancestor::div[contains(@class,"wpb_wrapper")]'
        )

        if iqac_div:
            text_content = iqac_div.xpath('.//p/text()').get()
            if text_content:
                self.total_internal_quality_assurance_data['iqac_description'] = text_content.strip()
            else:
                self.log("No <p> tag or text content found within the IQAC div.")
        else:
            self.log("IQAC div not found. Check the XPath.")
    
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_internal_quality_assurance_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_internal_quality_assurance_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_FacilitiesSpider(scrapy.Spider):
    name = "facilities_spider"
    start_urls = ["https://ceconline.edu/about/facilities/"]

    total_facility_data = {}

    def parse(self, response):
        facility_description = response.xpath('//h2[text()="Facilities"]/following-sibling::div//p/text()').get()
        if facility_description:
            self.total_facility_data['facility_description'] = facility_description.strip()
        else:
            self.log("Facility description not found.")
    
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_facility_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}  # Reset if the file contains invalid data
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_facility_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)

class CEC_CollegeInfoSpider(scrapy.Spider):
    name = "college_info_spider1"
    start_urls = ["https://ceconline.edu/contact-us/"]

    total_college_contact_info_data = {}

    def parse(self, response):
        # Flatten contact details
        contact_items = response.xpath('//div[contains(@class, "stm-map__content")]//ul[@class="stm-contact-details__items"]/li')
        for item in contact_items:
            detail_text = item.xpath('.//text()').get().strip()
            detail_type = item.xpath('@class').get()
            if detail_type:
                detail_type = detail_type.split("_")[-1]
                self.total_college_contact_info_data[f'contact_{detail_type}'] = detail_text

        # Flatten location details
        location_section = response.xpath('//div[contains(@class, "entry-content")]//div[@class="wpb_wrapper"]')
        if location_section:
            main_description = location_section.xpath('./p[1]/text()').get("").strip()
            if main_description:
                self.total_college_contact_info_data['location_main_description'] = main_description

            # Flatten transit details
            transit_paragraphs = location_section.xpath('./p[position() > 1]')
            for i, p in enumerate(transit_paragraphs, 1):
                strong_text = p.xpath('./strong/text()').get("").strip()
                if strong_text:
                    rest_of_text = "".join(p.xpath('./text()[not(ancestor::strong)]').getall()).strip()
                    key = strong_text.replace(":", "").strip().lower().replace(" ", "_")
                    self.total_college_contact_info_data[f'transit_distance_{key}'] = rest_of_text

        # Flatten Google Maps info
        google_maps_card = response.xpath('//div[contains(@class, "place-card") and contains(@jsaction, "placeCard.directions")]')
        if google_maps_card:
            link = google_maps_card.xpath('.//a[contains(@class, "navigate-link")]/@href').get()
            if link:
                college_name = link.split("+")[0].replace("College+of+Engineering+", "").strip()
                self.total_college_contact_info_data['maps_college_name'] = college_name
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_college_contact_info_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_college_contact_info_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)            

class CEC_CommitteeSpider(scrapy.Spider):
    name = "committee_spider1"
    start_urls = ["https://ceconline.edu/about/committees/rti/"]

    total_committee_data = {}

    def parse(self, response):
        committees_xpath = response.xpath(
            '//div[contains(@class, "vc_column-inner")]'
            '/div[contains(@class, "wpb_wrapper")]'
            '/div[contains(@class, "wpb_text_column") and contains(@class, "wpb_content_element")]'
        )

        committee_count = 1
        for committee_section in committees_xpath:
            committee_title = committee_section.xpath(".//h3/strong/text()").get()
            if committee_title:
                committee_title = committee_title.strip()
                # self.total_committee_data[f'committee_{committee_count}_name'] = committee_title
                
                officer_info = committee_section.xpath(".//p")
                officer_count = 1
                for officer in officer_info:
                    text_parts = officer.xpath(".//span/text()").getall()
                    if text_parts:
                        name = text_parts[0].strip()
                        designation = text_parts[1].strip() if len(text_parts) > 1 else None
                        contact = officer.xpath('.//a/@href').get()
                        
                        base_key = f'committee_{committee_count}_{committee_title}_officer_{officer_count}'
                        self.total_committee_data[f'{base_key}_name'] = name
                        if designation:
                            self.total_committee_data[f'{base_key}_designation'] = designation
                        if contact:
                            self.total_committee_data[f'{base_key}_contact'] = contact
                        
                        officer_count += 1
                
                committee_count += 1
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_committee_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_committee_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)            

class CEC_BoardOfGovernorsSpider(scrapy.Spider):
    name = "bog1"
    start_urls = ['https://ceconline.edu/board-of-governors/']

    total_bog_data = {}
    
    def parse(self, response):
        table = response.xpath('//div[@class="wpb_wrapper"]//table[contains(., "Board of Governors")]')
        rows = table.xpath('.//tr[td]')

        for i, row in enumerate(rows, 1):
            name_designation = row.xpath('td[2]/strong/text()').get()
            designation = row.xpath('td[3]/strong/text()').get()
            # role = row.xpath('td[4]/strong/text()').get()
            
            if name_designation and designation:
                self.total_bog_data[f'bog_member_{i}_name_designation'] = name_designation.strip()
                self.total_bog_data[f'bog_member_{i}_designation'] = designation.strip()
                # if role:
                #     self.total_bog_data[f'bog_member_{i}_role'] = role.strip()
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_bog_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_bog_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)                

class CEC_AntiRaggingSpider(scrapy.Spider):
    name = "antiragging_spider1"
    start_urls = ["https://ceconline.edu/about/committees/anti_ragging/"]

    total_anti_ragging_cell_data = {}

    def parse(self, response):
        antiragging_section = response.xpath(
            '//div[contains(@class, "wpb_text_column") and contains(@class, "wpb_content_element") and contains(@class,"vc_custom_")]'
        )

        if antiragging_section:
            for i, li in enumerate(antiragging_section.xpath('.//ul/li'), 1):
                function_text = li.xpath('.//text()').get("").strip()
                if function_text:
                    self.total_anti_ragging_cell_data[f'anti_ragging_function_{i}'] = function_text
        else:
            self.log("Anti-ragging section not found. Check the XPath.")

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_anti_ragging_cell_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_anti_ragging_cell_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)
class CEC_AnnounceSpider(scrapy.Spider):
    name = 'announce_spider'
    start_urls = ['https://ceconline.edu/announcements/']

    total_announcement_data = {}
    
    def parse(self, response):
        notice_mapping = {
            '0fa72ab': {'id': 'btech_registration', 'text': 'Registration Form - B.Tech Admission 2024'},
            '7089465': {'id': 'btech_fee', 'text': 'B.Tech  Fee Structure 2024-25'},
            '2099742': {'id': 'nri_fee', 'text': 'NRI Fee Structure'},
            'd2d9dc9': {'id': 'let_fee', 'text': 'LET Fee Structure'},
            '4b5f897': {'id': 'mca_fee', 'text': 'MCA Fee Structure'}
        }
        
        for element_id, info in notice_mapping.items():
            notice_element = response.xpath(
                f"//div[contains(@class, 'elementor-element-{element_id}')]"
                "//div[contains(@class, 'elementor-widget-container')]"
                "//h2[contains(@class, 'elementor-heading-title')]"
                "//a"
            ).get() 
            
            if notice_element:
                notice_selector = scrapy.Selector(text=notice_element)
                link = notice_selector.xpath('//a/@href').get('')
                text = notice_selector.css('u::text').get('')
                
                if not text: 
                    text = notice_selector.xpath('//a/text()').get('').strip()
                if link and link.startswith('/'):
                    link = urljoin(response.url, link)
                    
                if link:  
                    doc_type = self.get_document_type(link)
                    
                    # Flatten the announcement data
                    # self.total_announcement_data[f'announcement_{info["id"]}_title'] = text
                    self.total_announcement_data[f'announcement_{info["id"]}_{text}_url'] = link
                    # self.total_announcement_data[f'announcement_{info["id"]}_{text}_type'] = doc_type
                    # self.total_announcement_data[f'announcement_{info["id"]}_{text}_element_id'] = element_id
                    
                    if doc_type == 'pdf':
                        yield scrapy.Request(
                            url=link,
                            callback=self.save_pdf,
                            meta={'title': text, 'notice_id': info['id']},
                            errback=self.handle_error
                        )

    def get_document_type(self, url):
        if url.endswith('.pdf'):
            return 'pdf'
        elif 'forms.gle' in url:
            return 'google_form'
        elif url.startswith('http') or url.startswith('https'):
            return 'external_link'
        return 'other'
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_announcement_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_announcement_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)
class CEC_AICTEFeedbackSpider(scrapy.Spider):
    name = "aicte_feedback_spider"
    start_urls = ["https://ceconline.edu/aicte-feedback/"]

    total_aicte_data = {}

    def parse(self, response):
        notice_div = response.xpath(
            '//div[contains(@class, "wpb_wrapper")]'
            '/h1[contains(text(), "NOTICE")]'
            '/following-sibling::node()' 
        )
        if notice_div:
            link_count = 1
            for sibling in notice_div:
                if sibling.xpath('self::p'):
                    links = sibling.xpath(".//a")
                    for link in links:
                        text = link.xpath(".//text()").get()
                        href = link.xpath("@href").get()
                        if text and href:
                            # self.total_aicte_data[f'aicte_feedback_link_{link_count}_text'] = text.strip()
                            self.total_aicte_data[f'aicte_feedback_link_{link_count}_url'] = href.strip()
                            link_count += 1
        else:
            self.log("Notice div 'NOTICE' not found. Check the XPath.")
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_aicte_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_aicte_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)        

class CEC_AdminStaffSpider(scrapy.Spider):
    name = "admin_staff"
    start_urls = ["https://ceconline.edu/about/administration/administrative-staff/"]

    total_admin_staff_data = {}

    def parse(self, response):
        table = response.xpath("//h2[text()='Administrative Staff']/following-sibling::div//table")

        if not table:
            self.logger.error("Table not found. Verify the XPath selector.")
            return

        rows = table.xpath(".//tr")
        for i, row in enumerate(rows[1:], 1):  # Skip header row
            name = row.xpath(".//td[1]//text()").get()
            designation = row.xpath(".//td[2]//text()").get()

            if name and designation:
                # self.total_admin_staff_data[f'admin_staff_{i}_name'] = name.strip()
                self.total_admin_staff_data[f'admin_staff_{i}_name and designation'] = name.strip(),designation.strip()
    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_admin_staff_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_admin_staff_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)            

class CEC_nri_AdmissionSpider(scrapy.Spider):
    name = 'nri_admission_spider'
    start_urls = ['https://ceconline.edu/b-tech-nri-admission-2024-25/']
    
    total_admission_data = {}

    def parse(self, response):
        admission_links = {
            "candidate_details": "//div[@data-id='db33835']//a/@href",
            "original_certificates": "//div[@data-id='7e372bc']//a/@href",
            "admission_schedule": "//div[@data-id='cd4f4f3']//a/@href"
        }
        
        # Flatten PDF links and details
        for key in admission_links:
            link = response.xpath(admission_links[key]).get()
            if link:
                absolute_url = urljoin(response.url, link)
                self.total_admission_data[f'nri_admission_{key}_url'] = absolute_url
                
                # Get corresponding text
                text = response.xpath(f"//div[@data-id='{key}']//a/u/text()").get()
                if text:
                    self.total_admission_data[f'nri_admission_{key}_text'] = text.strip()
                    

    def closed(self, reason):
        """This function is automatically called when the spider finishes execution."""
        self.logger.info(json.dumps(self.total_admission_data, indent=4))

        # Check if JSON file exists and load existing data
        if os.path.exists('college_json_data/cec.json'):
            try:
                with open('college_json_data/cec.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}

        # Merge and save updated data
        existing_data.update(self.total_admission_data)
        with open('college_json_data/cec.json', 'w') as f:
            json.dump(existing_data, f, indent=4)
       