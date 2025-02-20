from playwright.sync_api import sync_playwright
import json

def scrape_principal_details(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/administrations')

    button_selector = ".sidebar-nav-li:has-text('Principal')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.grid') 
    name = page.locator('.custom-a p').text_content()
    position = page.locator('.person-position').text_content()
    email = page.locator('.bio-contact-item a').get_attribute('href').replace('mailto:', '')
    # image_url = page.locator('.bio-img img').get_attribute('src')
    # profile_link = page.locator('.person-name').locator('..').get_attribute('href') 

    principal_details = {
        "Name of Principal": name,
        "Position of Principal": position,
        "Email of Principal": email,
        # "Image URL of Principal": image_url,
        # "Profile Link of Principal": profile_link,
    }

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(principal_details, file, ensure_ascii=False, indent=4)

    browser.close()

def scrape_about_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/administrations')
    button_selector = ".sidebar-nav-li:has-text('About')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.about')
    about_text = page.locator('.about .about-content .custom-p').text_content()
    # about_details = {
    #     "About Model Engineering College": about_text
    # }  
    # try:
    #     with open("mec.json", "r", encoding="utf-8") as file:
    #         existing_data = json.load(file)
    # except FileNotFoundError:
    #     existing_data = {} 
    # existing_data.update(about_details)
    # with open("college_json_data/mec.json", "w", encoding="utf-8") as file:
    #     json.dump(existing_data, file, ensure_ascii=False, indent=4)
    # browser.close()

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    data["About Model Engineering College"] = about_text

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
def scrape_board_of_governors_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/administrations')
    button_selector = ".sidebar-nav-li:has-text('Board-of-Governers')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.board')
    # description = page.locator('.board .custom-p').text_content().replace("\r", "").replace("\n", "").strip(),
    members = []
    for member in page.locator('.photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().replace("\r", "").replace("\n", "").strip(),
            "Position": member.locator('.person-position').text_content().replace("\r", "").replace("\n", "").strip(),
            # "Description": member.locator('p:nth-child(4)').text_content().replace("\r", "").replace("\n", "").strip(),
            # "Image URL": member.locator('img').get_attribute('src').replace("\r", "").replace("\n", "").strip(),
        }
        members.append(member_details)

    # board_details = {
    #     "Description of board of governors": description,
    #     "Members of board of governors": members
    # } 
    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {} 
    # existing_data.update(board_details)
    # existing_data["Description of board of governors"] = description
    existing_data["Members of board of governors"] = members
    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    browser.close()

def scrape_administrative_staff_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/administrations')
    button_selector = ".sidebar-nav-li:has-text('Administrative Staff')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.board')
    page.wait_for_selector('.photo-item')
    staff_members = []
    for member in page.locator('.board .photo-item').all():
        member_details = {
            "Name of administrative staff": member.locator('.person-name').text_content().strip(),
            "Position of administrative staff": member.locator('.person-position').text_content().strip(),
            # "Image URL of administrative staff": member.locator('img').get_attribute('src'),
            # Some members have additional description (like "On deputation")
            # "Additional Info of administrative staff": member.locator('p:nth-child(4)').text_content().strip() if member.locator('p:nth-child(4)').count() > 0 else ""
        }
        staff_members.append(member_details)

    # admin_staff_details = {
    #     "Administrative_Staff of Model Engineering College": staff_members
    # } 
    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {} 
    # existing_data.update(admin_staff_details)
    existing_data["Members of administrative staff"] = member_details
    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    browser.close()

def scrape_academic_council_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/administrations')
    button_selector = ".sidebar-nav-li:has-text('Academic Council')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.grid')
    # description = page.locator('.grid > .custom-p').text_content()
    
    members = []
    for member in page.locator('.grid .photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().strip(),
            "Position": member.locator('.person-position').text_content().strip(),
            # "Description": member.locator('p:nth-child(4)').text_content().strip() if member.locator('p:nth-child(4)').count() > 0 else "",
            # "Image URL": member.locator('img').get_attribute('src')
        }
        members.append(member_details)

    functions = []
    for function in page.locator('.function-list ul li').all():
        functions.append(function.text_content().strip())

    # academic_council_details = {
    #     "Description of Academic Council": description,
    #     "Members of Academic Council": members,
    #     "Functions of Academic Council": functions
    # } 
    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {} 
    # existing_data.update(academic_council_details)
    # existing_data["Description of Academic Council"] = description
    existing_data["Members of Academic Council"] = members
    existing_data["Functions of Academic Council"] = functions

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    browser.close()

def scrape_pta_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/administrations')
    button_selector = ".sidebar-nav-li:has-text('PTA')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.page-content')
    description = page.locator('.page-content .custom-p').text_content()
    members = []
    for member in page.locator('.list-image-container .photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().strip(),
            "Position": member.locator('.person-position').text_content().strip(),
            # "Image URL": member.locator('img').get_attribute('src')
        }
        members.append(member_details)

    # pta_details = {
    #     "Description of PTA": description,
    #     "Members of PTA": members
    # } 
    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {} 
    # existing_data.update(pta_details)
    existing_data["Description of PTA"] = description
    existing_data["Members of PTA"] = members
    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    browser.close()

def scrape_senate_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/administrations')
    button_selector = ".sidebar-nav-li:has-text('Senate')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.Senate')
    # description = page.locator('.Senate > .custom-p').text_content()
    members = []
    for member in page.locator('.Senate .photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().strip(),
            "Position": member.locator('.person-position').text_content().strip(),
            # "Additional Info": member.locator('p:nth-child(4)').text_content().strip() if member.locator('p:nth-child(4)').count() > 0 else "",
            # "Image URL": member.locator('img').get_attribute('src')
        }
        members.append(member_details)

    # senate_details = {
    #     "Description of Senate": description,
    #     "Members of Senate": members
    # } 
    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {} 
    # existing_data.update(senate_details)
    # existing_data["Description of Senate"] = description
    existing_data["Members of Senate"] = members
    
    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    browser.close()



def scrape_admission_details_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/admissions2024')
    page.wait_for_selector('.admission')
    admission_procedure = page.locator('.admission p').text_content()

    ug_programmes = []
    for card in page.locator('.admissioncardholder .admissionscard').all():
        programme = {
            f"{card.locator('.course-admission2k23').text_content().strip()} Programme": card.locator('h3').text_content().strip(),
            # "Course": card.locator('.course-admission2k23').text_content().strip(),
            f"Seats for {card.locator('h3').text_content().strip()}": card.locator('.seats-admission2k23').text_content().strip(),
            # "Is New": True if card.locator('#newbadge').count() > 0 else False
        }
        ug_programmes.append(programme)
    fee_structure = []
    for row in page.locator('.fee-table tbody tr').all():
        fee = {
            "Category": row.locator('td:nth-child(1)').text_content().strip(),
            "Amount": row.locator('td:nth-child(2)').text_content().strip()
        }
        fee_structure.append(fee)
    seat_matrix = {
        "Merit Regulated": "50%",
        "Merit Full Fees": "45%",
        "NRI": "5%"
    }

    pg_programmes = []
    for card in page.locator('.admissioncardholder:nth-child(2) .admissionscard').all():
        programme = {
            f"{card.locator('.course-admission2k23').text_content().strip()} Programme": card.locator('h3').text_content().strip(),
            # "Course": card.locator('.course-admission2k23').text_content().strip(),
            f"Seats for {card.locator('h3').text_content().strip()}": card.locator('.seats-admission2k23').text_content().strip()
        }
        pg_programmes.append(programme)

    # admission_details = {
    #     "Admission_Procedure": admission_procedure,
    #     "UG_Programmes": ug_programmes,
    #     "Fee_Structure": fee_structure,
    #     "Seat_Matrix": seat_matrix,
    #     "PG_Programmes": pg_programmes
    # } 
    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {} 
    # existing_data.update(admission_details)
    existing_data["Admission_Procedure"] = admission_procedure
    existing_data["UG_Programmes"] = ug_programmes
    existing_data["Fee_Structure"] = fee_structure
    existing_data["Seat_Matrix"] = seat_matrix
    existing_data["PG_Programmes"] = pg_programmes
    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    browser.close()

def scrape_facilities(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/facilities')
    
    facilities = []

    for facility in page.locator('.facilities-page-items .gen-facility-item').all():
        facility_details = {
            "Name": facility.locator('.custom-h3').text_content().strip().replace('\xa0', ''),
            f"{facility.locator('.custom-h3').text_content().strip().replace('\xa0', '')} Description": facility.locator('.custom-p').text_content().strip(),
            # "Icon URL": facility.locator('.facility-icon').get_attribute('src'),
            # "Link": facility.locator('a').get_attribute('href') if facility.locator('a').count() > 0 else None
        }
        facilities.append(facility_details)
    library_details = {
        "Name": "Library",
        "Library Description": page.locator('.page-content > div:last-child > div > .custom-p').text_content().strip(),
        # "Icon URL": page.locator('img[alt="library"]').get_attribute('src'),
        # "Link": page.locator('.custom-a').get_attribute('href')
    }
    facilities.append(library_details)

    # facilities_details = {
    #     "Facilities in Model Engineering College": facilities
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(facilities_details)
    existing_data["Facilities in Model Engineering College"] = facilities

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()

def scrape_about_statutory_committee(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/committees')
    button_selector = ".sidebar-nav-li:has-text('About')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.about')
    about_text = page.locator('.about .custom-p').text_content()
    
    # about_details = {
    #     "About_Statutory_Committees": about_text
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(about_details)
    existing_data["About_Statutory_Committees"] = about_text

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()

def scrape_iqac_section(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/committees')
    button_selector = ".sidebar-nav-li:has-text('Internal Quality Assurance Cell')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.grid')
    description = page.locator('.grid > .custom-p').text_content()
    
    functions = []
    for item in page.locator('.grid ul li').all():
        functions.append(item.text_content().strip())
    
    # meeting_minutes = []
    # for link in page.locator('.grid > a').all():
    #     meeting_minutes.append({
    #         "Title": link.text_content().strip(),
    #         "URL": link.get_attribute('href')
    #     })
    
    members = []
    for member in page.locator('.list-image-container .photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().strip(),
            "Position": member.locator('.person-position').text_content().strip(),
            # "Additional Info": member.locator('p:nth-child(4)').text_content().strip() if member.locator('p:nth-child(4)').count() > 0 else "",
            # "Image URL": member.locator('img').get_attribute('src')
        }
        members.append(member_details)

    # iqac_details = {
    #     "Description of Internal Quality Assurance Cell": description,
    #     "Functions of Internal Quality Assurance Cell": functions,
    #     "Meeting Minutes of Internal Quality Assurance Cell": meeting_minutes,
    #     "Members of Internal Quality Assurance Cell": members
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(iqac_details)
    existing_data["Description of Internal Quality Assurance Cell"] = description
    existing_data["Functions of Internal Quality Assurance Cell"] = functions
    # existing_data["Meeting Minutes of Internal Quality Assurance Cell"] = meeting_minutes
    existing_data["Members of Internal Quality Assurance Cell"] = members

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()

def scrape_grievance_cell(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/committees')
    button_selector = ".sidebar-nav-li:has-text('Grievance Cell')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.about')
    description = page.locator('.about > .custom-p').text_content()
    
    members = []
    for member in page.locator('.list-image-container .photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().strip(),
            "Position": member.locator('.person-position').text_content().strip(),
            # "Image URL": member.locator('img').get_attribute('src')
        }
        members.append(member_details)

    # grievance_cell_details = {
    #     "Description of grievance cell": description,
    #     "Members of grievance cell": members
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(grievance_cell_details)
    existing_data["Description of grievance cell"] = description
    existing_data["Members of grievance cell"] = members

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()


def scrape_anti_ragging_committee(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/committees')
    button_selector = ".sidebar-nav-li:has-text('Anti-ragging Committee')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.about')
    description = page.locator('.about > .custom-p').first.text_content()
    
    # Get references
    # references = []
    # for item in page.locator('.about ol li').all():
    #     references.append(item.text_content().strip())
    
    # Get committee members
    members = []
    members.append({
        "Position": "Chairman",
        "Name": "Principal"
    })
    members.append({
        "Position": "Member Secretary",
        "Name": "Dr. Sreenivas P, Associate Prof. in Mechanical (Chairman – Anti ragging squad.)"
    })

    # anti_ragging_committee_details = {
    #     "Description of anti_ragging_committee_details": description,
    #     "References of anti_ragging_committee_details": references,
    #     "Members of anti_ragging_committee_details": members
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(anti_ragging_committee_details)
    existing_data["Description of anti_ragging_committee_details"] = description
    # existing_data["References of anti_ragging_committee_details"] = references
    existing_data["Members of anti_ragging_committee_details"] = members

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()


def scrape_anti_ragging_squad(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/committees')
    button_selector = ".sidebar-nav-li:has-text('Anti-Ragging Squad')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.about')
    description = page.locator('.about > .custom-p').first.text_content()
    
    references = []
    for item in page.locator('.about ol li').all():
        references.append(item.text_content().strip())
    
    members = []
    for member in page.locator('.list-image-container .photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().strip(),
            "Position": member.locator('.person-position').text_content().strip(),
            # "Additional Info": member.locator('p:nth-child(4)').text_content().strip() if member.locator('p:nth-child(4)').count() > 0 else "",
            # "Image URL": member.locator('img').get_attribute('src')
        }
        members.append(member_details)

    # anti_ragging_squad_details = {
    #     "Description of anti ragging squad": description,
    #     "References of anti ragging squad": references,
    #     "Members of anti ragging squad": members
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(anti_ragging_squad_details)
    existing_data["Description of anti ragging squad"] = description
    # existing_data["References of anti ragging squad"] = references
    existing_data["Members of anti ragging squad"] = members

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()


def scrape_anti_sexual_harassment_cell(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/committees')
    button_selector = ".sidebar-nav-li:has-text('Anti-sexual Harassment & Internal Compliance Cell')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.about')
    descriptions = page.locator('.about > p.custom-p').all()
    main_description = "\n".join([desc.text_content().strip() for desc in descriptions[:2]])
    
    # Get objectives
    objectives = []
    for item in page.locator('.about > ul').first.locator('li').all():
        objectives.append(item.text_content().strip())
    
    # Get definition of sexual harassment
    harassment_definition = page.locator('.about > p.custom-p').nth(2).text_content()
    harassment_types = []
    for item in page.locator('.about > ul').nth(1).locator('li').all():
        harassment_types.append(item.text_content().strip())
    
    # Get committee members
    members = []
    for member in page.locator('.list-image-container .photo-item').all():
        member_details = {
            "Name": member.locator('.person-name').text_content().strip(),
            # "Position": member.locator('.person-position').text_content().strip(),
            "Contact": member.locator('p:nth-child(4)').text_content().strip() if member.locator('p:nth-child(4)').count() > 0 else "",
            # "Image URL": member.locator('img').get_attribute('src')
        }
        members.append(member_details)
    
    # Get complaint handling info
    complaint_info = page.locator('.about > p.custom-p').nth(-2).text_content()
    false_reporting_info = page.locator('.about > p.custom-p').nth(-1).text_content()
    
    # Get links
    handbook_link = page.locator('a[href*="Anti-Sexual-Harrassment-Handbook.pdf"]').get_attribute('href')
    complaint_form_link = page.locator('a[href*="forms.gle"]').get_attribute('href')

    # cell_details = {
    #     "Description of Anti-sexual Harassment & Internal Compliance Cell": main_description,
    #     "Objectives of Anti-sexual Harassment & Internal Compliance Cell": objectives,
    #     "Sexual_Harassment_Definition": harassment_definition,
    #     "Types_of_Sexual_Harassment": harassment_types,
    #     "Members of Anti-sexual Harassment & Internal Compliance Cell": members,
    #     "Complaint_Handling of Anti-sexual Harassment & Internal Compliance Cell": complaint_info,
    #     "False_Reporting": false_reporting_info,
    #     "Handbook_Link": handbook_link,
    #     "Complaint_Form_Link of Anti-sexual Harassment & Internal Compliance Cell": complaint_form_link
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(cell_details)
    existing_data["Description of Anti-sexual Harassment & Internal Compliance Cell"]= main_description
    existing_data["Objectives of Anti-sexual Harassment & Internal Compliance Cell"]= objectives
    # existing_data["Sexual_Harassment_Definition"]= harassment_definition
    # existing_data["Types_of_Sexual_Harassment"]= harassment_types
    existing_data["Members of Anti-sexual Harassment & Internal Compliance Cell"]= members
    existing_data["Complaint_Handling of Anti-sexual Harassment & Internal Compliance Cell"]= complaint_info
    existing_data["False_Reporting"]= false_reporting_info
    existing_data["Handbook_Link"]= handbook_link
    existing_data["Complaint_Form_Link of Anti-sexual Harassment & Internal Compliance Cell"]= complaint_form_link

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()

def scrape_safety_manual(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/committees')
    button_selector = ".sidebar-nav-li:has-text('Safety Manual')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    page.wait_for_selector('.page-content .about', timeout=5000)
    description = page.locator('.page-content .about p.custom-p').first.text_content().strip()   
    manual_link = page.locator('a[href*="safety_manual.pdf"]').get_attribute('href')
    # safety_manual_details = {
    #     "Description of safety manual": description,
    #     "Safety Manual Download Link": manual_link
    # }

    try:
        with open("mec.json", "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # existing_data.update(safety_manual_details)
    existing_data["Description of safety manual"] = description
    existing_data["Safety Manual Download Link"] = manual_link

    with open("mec.json", "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()
def comp_sci_section(playwright, section_name, output_file="mec.json"):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/departments/cse')

    button_selector = f".sidebar-nav-li:has-text('{section_name}')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)

    flat_data = {}

    if section_name == "About":
        page.wait_for_selector('.about')
        flat_data[f"{section_name} Computer Science department Description"] = page.locator('.about > .custom-p').text_content().strip()

    # elif section_name == "Vision & Mission":
    #     page.wait_for_selector('.vision-mission')

    #     flat_data["Vision"] = page.locator('.vision .custom-p').text_content().strip()

    #     mission_items = page.locator('.mission-item').all()
    #     for i, item in enumerate(mission_items, start=1):
    #         paragraphs = item.locator('.custom-p').all()
    #         if len(paragraphs) >= 2:
    #             mission_number = paragraphs[0].text_content().strip()
    #             mission_text = paragraphs[1].text_content().strip()
    #             flat_data[f"COmputer Science department Mission_{mission_number}"] = mission_text

    #     peo_items = page.locator('h2:text("Program Educational Objectives") ~ .ed-obj .ed-obj-item').all()
    #     for i, peo in enumerate(peo_items, start=1):
    #         if peo.locator('.custom-h3').count() > 0 and peo.locator('.custom-p').count() > 0:
    #             title = peo.locator('.custom-h3').text_content().strip()
    #             description = peo.locator('.custom-p').text_content().strip()
    #             flat_data[f"Program Educational Objectives of Computer science {i}_Title"] = title
    #             flat_data[f"Program Educational Objectives of Computer science{i}_Description"] = description

    #     pso_items = page.locator('h2:text("Program Specific Outcomes") ~ .ed-obj .ed-obj-item').all()
    #     for i, pso in enumerate(pso_items, start=1):
    #         if pso.locator('.custom-h3').count() > 0 and pso.locator('.custom-p').count() > 0:
    #             title = pso.locator('.custom-h3').text_content().strip()
    #             description = pso.locator('.custom-p').text_content().strip()
    #             flat_data[f"Program Specific Outcomes of Computer science{i}_Title"] = title
    #             flat_data[f"Program Specific Outcomes of Computer science{i}_Description"] = description

    #     po_items = page.locator('h2:text("Program Outcomes") ~ .ed-obj .ed-obj-item').all()
    #     for i, po in enumerate(po_items, start=1):
    #         if po.locator('.custom-h3').count() > 0 and po.locator('.custom-p').count() > 0:
    #             title = po.locator('.custom-h3').text_content().strip()
    #             description = po.locator('.custom-p').text_content().strip()
    #             flat_data[f"Program Outcomes of Computer science{i}_Title"] = title
    #             flat_data[f"Program Outcomes of Computer science{i}_Description"] = description

    elif section_name == "Courses Offered":
        page.wait_for_selector('.courses-offered')
        course_items = page.locator('.course-item').all()
        for i, item in enumerate(course_items, start=1):
            degree = item.locator('.custom-h2').text_content().strip()
            program = item.locator('.custom-h2').text_content().strip() + " " + item.locator('.custom-h3.red').first.text_content().strip()
            # flat_data[f"Course_{i}_Degree"] = degree
            flat_data[f"Course_{i}_Program"] = program
            if item.locator('.custom-h3.red').count() > 1:
                specialization = item.locator('.custom-h3.red').nth(1).text_content().strip()
                flat_data[f"Course_{i}_Specialization of Computer science"] = specialization

    elif section_name == "HOD":
        page.wait_for_selector('.grid')
        flat_data["HOD_Name of Computer science"] = page.locator('.photo-item .person-name').text_content().strip()
        flat_data["HOD_Position of Computer science"] = page.locator('.photo-item .person-position').text_content().strip()
        flat_data["HOD_Email of Computer science"] = page.locator('.bio-contact-item a').get_attribute('href').replace('mailto:', '')
        # flat_data["HOD_Image_URL of Computer science"] = page.locator('.photo-item img').get_attribute('src')

    elif section_name == "Faculty":
        page.wait_for_selector('.grid')
        faculty_members = page.locator('.photo-item').all()
        for i, member in enumerate(faculty_members, start=1):
            flat_data[f"Faculty_{i}_Name of Computer science"] = member.locator('.person-name').text_content().strip()
            flat_data[f"Faculty_{i}_Position of Computer science"] = member.locator('.person-position').text_content().strip()
            # flat_data[f"Faculty_{i}_Image_URL of Computer science"] = member.locator('img').get_attribute('src')
    elif section_name == "Associations":
        page.wait_for_selector('.asc')
        flat_data["association_name_of_applied_science"] = page.locator('.asc .custom-h3').text_content().strip()
        flat_data["association_description_of_applied_science"] = page.locator('.asc .custom-p').text_content().strip()
        
    elif section_name == "Facilities":
        page.wait_for_selector('.page-content')
        main_description = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        # flat_data["Facilities_Main_Description of Computer science"] = main_description

        facility_items = page.locator('.facility-items > div').all()
        for i, facility in enumerate(facility_items, start=1):
            name = facility.locator('.custom-h3').text_content().strip()
            description = facility.locator('.custom-p').text_content().strip()
            flat_data[f"Facility_{i}_Name of Computer science"] = name
            # flat_data[f"Facility_{i}_Description of Computer science"] = description

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    existing_data.update(flat_data)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()

def electronics_and_communication_section(playwright, section_name, output_file="mec.json"):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/departments/ece')

    button_selector = f".sidebar-nav-li:has-text('{section_name}')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)

    section_data = {}

    if section_name == "About":
        page.wait_for_selector('.about')
        section_data = {
            "Description_About electronics and communication department": page.locator('.about > .custom-p').text_content().strip()
        }

    # elif section_name == "Vision & Mission":
    #     page.wait_for_selector('.vision-mission')

    #     section_data["Vision of electronics and communication department"] = page.locator('.vision .custom-p').text_content().strip()

    #     mission_items = page.locator('.mission-item')
    #     for i, item in enumerate(mission_items.all(), start=1):
    #         paragraphs = item.locator('.custom-p').all()
    #         if len(paragraphs) >= 2:
    #             mission_number = paragraphs[0].text_content().strip()
    #             mission_text = paragraphs[1].text_content().strip()
    #             section_data[f"Mission_{mission_number} of electronics and communication department"] = mission_text

    #     peo_items = page.locator('h2:text("Program Educational Objectives") ~ .ed-obj .ed-obj-item')
    #     for i, peo in enumerate(peo_items.all(), start=1):
    #         if peo.locator('.custom-h3').count() > 0 and peo.locator('.custom-p').count() > 0:
    #             title = peo.locator('.custom-h3').text_content().strip()
    #             description = peo.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Educational Objectives of electronics and communication department {title} "] = description

    #     pso_items = page.locator('h2:text("Program Specific Outcomes") ~ .ed-obj .ed-obj-item')
    #     for i, pso in enumerate(pso_items.all(), start=1):
    #         if pso.locator('.custom-h3').count() > 0 and pso.locator('.custom-p').count() > 0:
    #             title = pso.locator('.custom-h3').text_content().strip()
    #             description = pso.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Specific Outcomes of electronics and communication department {title} "] = description

    #     po_items = page.locator('h2:text("Program Outcomes") ~ .ed-obj .ed-obj-item')
    #     for i, po in enumerate(po_items.all(), start=1):
    #         if po.locator('.custom-h3').count() > 0 and po.locator('.custom-p').count() > 0:
    #             title = po.locator('.custom-h3').text_content().strip()
    #             description = po.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Outcomes of electronics and communication department {title}"] = description

    elif section_name == "Courses Offered":
        page.wait_for_selector('.courses-offered')
        course_items = page.locator('.course-item')
        for i, item in enumerate(course_items.all(), start=1):
            degree = item.locator('.custom-h2').text_content().strip()
            program = f"{degree} {item.locator('.custom-h3.red').first.text_content().strip()}"
            specialization = ""
            if item.locator('.custom-h3.red').count() > 1:
                specialization = item.locator('.custom-h3.red').nth(1).text_content().strip()
                program += f" with specialization in {specialization}"

            #section_data[f"Course_{i}_Degree of electronics and communication department"] = degree
            section_data[f"Course_{i}_Program of electronics and communication department"] = program

    elif section_name == "HOD":
        page.wait_for_selector('.grid')
        page.wait_for_selector('.photo-item .person-name', state='visible')
        page.wait_for_selector('.photo-item .person-position', state='visible')
        section_data = {
            "HOD_Name of electronics and communication department": page.locator('.photo-item .person-name').text_content().strip(),
            "HOD_Position of electronics and communication department": page.locator('.photo-item .person-position').text_content().strip(),
            "HOD_Email of electronics and communication department": page.locator('.bio-contact-item a').get_attribute('href').replace('mailto:', '') if page.locator('.bio-contact-item a').count() > 0 else None,
            # "HOD_Image_URL of electronics and communication department": page.locator('.photo-item img').get_attribute('src'),
        }
        for key, value in section_data.items():
            section_data[key] = value
    elif section_name == "Faculty":
        page.wait_for_selector('.grid')
        faculty_items = page.locator('.photo-item')
        for i, member in enumerate(faculty_items.all(), start=1):
            name = member.locator('.person-name').text_content().strip()
            position = member.locator('.person-position').text_content().strip()
            image_url = member.locator('img').get_attribute('src')

            section_data[f"Faculty_{i}_Name of electronics and communication department"] = name
            section_data[f"Faculty_{i}_Position of electronics and communication department"] = position
            # section_data[f"Faculty_{i}_Image_URL of electronics and communication department"] = image_url

    elif section_name == "Facilities":
        page.wait_for_selector('.page-content')
        # section_data["Facilities_Main_Description"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()

        facility_items = page.locator('.facility-items > div')
        for i, facility in enumerate(facility_items.all(), start=1):
            name = facility.locator('.custom-h3').text_content().strip()
            description = facility.locator('.custom-p').text_content().strip()

            section_data[f"Facility_{i}_Name of electronics and communication department"] = name
            # section_data[f"Facility_{i}_Description of electronics and communication department"] = description

    # elif section_name == "Resources":
    #     page.wait_for_selector('.page-content')
    #     section_data["Resources_Main_Description of electronics and communication department"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()

    #     resource_items = page.locator('.res > div')
    #     for i, resource in enumerate(resource_items.all(), start=1):
    #         title = resource.locator('.custom-h3').text_content().strip()
    #         description = resource.locator('.custom-p').text_content().strip()

    #         section_data[f"Resource_{i}_Title of electronics and communication department"] = title
    #         section_data[f"Resource_{i}_Description of electronics and communication department"] = description

    #         links = resource.locator('.custom-a').all()
    #         for j, link in enumerate(links, start=1):
    #             link_text = link.text_content().strip()
    #             link_url = link.get_attribute('href')
    #             section_data[f"Resource_{i}_Link_{j}_Text of electronics and communication department"] = link_text
    #             section_data[f"Resource_{i}_Link_{j}_URL of electronics and communication department"] = link_url

    elif section_name == "Achievements":
        page.wait_for_selector('.page-content')
        # section_data["Achievements_Main_Description of electronics and communication department"] = page.locator('.page-content > div > div > .custom-p').text_content().strip()

        achievement_items = page.locator('.std-achievements ul > li')
        for i, item in enumerate(achievement_items.all(), start=1):
            achievement_text = item.inner_text().strip()
            section_data[f"Achievement of electronics and communication department {i}"] = achievement_text
    elif section_name == "Associations":
        page.wait_for_selector('.asc')
        section_data["association_name_of_applied_science"] = page.locator('.asc .custom-h3').text_content().strip()
        # section_data["association_description_of_applied_science"] = page.locator('.asc .custom-p').text_content().strip()

    elif section_name == "Recent Projects":
        page.wait_for_selector('.page-content')
        # section_data["Projects_Main_Description of electronics and communication department"] = page.locator('.page-content > div > div > .custom-p').first.text_content().strip()

        project_items = page.locator('.project-item')
        for i, project in enumerate(project_items.all(), start=1):
            title = project.locator('.custom-h3').text_content().strip()
            description = project.locator('.custom-p').text_content().strip()

            section_data[f"Project_{i}_Title of electronics and communication department"] = title
            section_data[f"Project_{i}_Description of electronics and communication department"] = description

            # links = project.locator('a').all()
            # for j, link in enumerate(links, start=1):
            #     link_text = link.text_content().strip()
            #     link_url = link.get_attribute('href')
            #     section_data[f"Project_{i}_Link_{j}_Text of electronics and communication department"] = link_text
            #     section_data[f"Project_{i}_Link_{j}_URL of electronics and communication department"] = link_url

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    existing_data.update(section_data)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    browser.close()
def electrical_and_electronics_section(playwright, section_name, output_file="mec.json"):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/departments/eee')

    # Increase timeout and add state='visible' to ensure element is clickable
    button_selector = f".sidebar-nav-li:has-text('{section_name}')"
    try:
        page.wait_for_selector(button_selector, state='visible', timeout=60000)
        page.click(button_selector)
    except Exception as e:
        print(f"Error finding selector '{button_selector}': {str(e)}")
        browser.close()
        return

    section_data = {}

    if section_name == "About":
        page.wait_for_selector('.about')
        section_data = {
            "Description of electrical and electronics department": page.locator('.about > .custom-p').text_content().strip()
        }

    # elif section_name == "Vision & Mission":
    #     page.wait_for_selector('.vision-mission')

    #     section_data["Vision of electrical and electronics department"] = page.locator('.vision .custom-p').text_content().strip()

    #     for item in page.locator('.mission-item').all():
    #         paragraphs = item.locator('.custom-p').all()
    #         if len(paragraphs) >= 2:
    #             mission_number = paragraphs[0].text_content().strip()
    #             mission_text = paragraphs[1].text_content().strip()
    #             section_data[f"Mission {mission_number} of electrical and electronics department"] = mission_text

    #     for peo in page.locator('h2:text("Program Educational Objectives") ~ .ed-obj .ed-obj-item').all():
    #         if peo.locator('.custom-h3').count() > 0 and peo.locator('.custom-p').count() > 0:
    #             peo_title = peo.locator('.custom-h3').text_content().strip()
    #             peo_desc = peo.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Educational Objective  of electrical and electronics department: {peo_title}"] = peo_desc

    #     for pso in page.locator('h2:text("Program Specific Outcomes") ~ .ed-obj .ed-obj-item').all():
    #         if pso.locator('.custom-h3').count() > 0 and pso.locator('.custom-p').count() > 0:
    #             pso_title = pso.locator('.custom-h3').text_content().strip()
    #             pso_desc = pso.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Specific Outcome  of electrical and electronics department: {pso_title}"] = pso_desc

    #     for po in page.locator('h2:text("Program Outcomes") ~ .ed-obj .ed-obj-item').all():
    #         if po.locator('.custom-h3').count() > 0 and po.locator('.custom-p').count() > 0:
    #             po_title = po.locator('.custom-h3').text_content().strip()
    #             po_desc = po.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Outcome  of electrical and electronics department: {po_title}"] = po_desc

    elif section_name == "Courses Offered":
        page.wait_for_selector('.courses-offered')
        for item in page.locator('.course-item').all():
            degree = item.locator('.custom-h2').text_content().strip()
            program = f"{degree} {item.locator('.custom-h3.red').first.text_content().strip()}"
            if item.locator('.custom-h3.red').count() > 1:
                specialization = item.locator('.custom-h3.red').nth(1).text_content().strip()
                program += f" with specialization in {specialization}"
            section_data[f"Course: {program} of electrical and electronics department"] = degree

    elif section_name == "HOD":
        page.wait_for_selector('.grid')
        section_data["HOD Name  of electrical and electronics department"] = page.locator('.photo-item .person-name').text_content().strip()
        section_data["HOD Position  of electrical and electronics department"] = page.locator('.photo-item .person-position').text_content().strip()
        section_data["HOD Email of electrical and electronics department"] = page.locator('.bio-contact-item a').get_attribute('href').replace('mailto:', '') if page.locator('.bio-contact-item a').count() > 0 else None
        # section_data["HOD Image URL of electrical and electronics department"] = page.locator('.photo-item img').get_attribute('src')

    elif section_name == "Faculty":
        page.wait_for_selector('.grid')
        for idx, member in enumerate(page.locator('.photo-item').all(), start=1):
            name = member.locator('.person-name').text_content().strip()
            position = member.locator('.person-position').text_content().strip()
            image_url = member.locator('img').get_attribute('src')
            section_data[f"Faculty {idx} Name of electrical and electronics department"] = name
            section_data[f"Faculty {idx} Position of electrical and electronics department"] = position
            # section_data[f"Faculty {idx} Image URL of electrical and electronics department"] = image_url

    elif section_name == "Facilities":
        page.wait_for_selector('.page-content')
        # section_data["Facilities Main Description"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        for facility in page.locator('.facility-items > div').all():
            name = facility.locator('.custom-h3').text_content().strip()
            desc = facility.locator('.custom-p').text_content().strip()
            section_data[f"Facility of electrical and electronics department: {name}"] = desc

    # elif section_name == "Resources":
    #     page.wait_for_selector('.page-content')
        # section_data["Resources Main Description"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        # for resource in page.locator('.res > div').all():
        #     name = resource.locator('.custom-h3').text_content().strip()
        #     desc = resource.locator('.custom-p').text_content().strip()
        #     section_data[f"Resource of electrical and electronics department: {name}"] = desc
            # for link in resource.locator('.custom-a').all():
            #     link_text = link.text_content().strip()
            #     link_url = link.get_attribute('href')
            #     section_data[f"Resource of electrical and electronics department {name} Link: {link_text}"] = link_url

    # elif section_name == "Achievements":
    #     try:
    #         page.wait_for_selector('.page-content')
    #         # section_data["Achievements Main Description"] = page.locator('.page-content > div > div > .custom-p').text_content().strip()
    #         for idx, item in enumerate(page.locator('.std-achievements ul > li').all(), start=1):
    #             achievement_text = item.inner_text().strip()
    #             section_data[f"Achievement {idx} of electrical and electronics department"] = achievement_text
    #     except Exception as e:
    #         print(f"Error scraping achievements: {str(e)}")

    # elif section_name == "Associations":
    #     page.wait_for_selector('.asc')
    #     section_data["association_name_of_applied_science"] = page.locator('.asc .custom-h3').text_content().strip()
    #     # section_data["association_description_of_applied_science"] = page.locator('.asc .custom-p').text_content().strip()

    # elif section_name == "Recent Projects":
    #     try:
    #         page.wait_for_selector('.page-content')
    #         # section_data["Projects Main Description of electrical and electronics department"] = page.locator('.page-content > div > div > .custom-p').first.text_content().strip()
    #         for idx, project in enumerate(page.locator('.project-item').all(), start=1):
    #             name = project.locator('.custom-h3').text_content().strip()
    #             desc = project.locator('.custom-p').text_content().strip()
    #             section_data[f"Project {idx} Name of electrical and electronics department"] = name
    #             section_data[f"Project {idx} Description of electrical and electronics department"] = desc
    #             for link in project.locator('a').all():
    #                 link_text = link.text_content().strip()
    #                 link_url = link.get_attribute('href')
    #                 section_data[f"Project {idx} of electrical and electronics department Link: {link_text}"] = link_url
    #     except Exception as e:
    #         print(f"Error scraping projects: {str(e)}")

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    
    existing_data.update(section_data)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    
    browser.close()

def electronics_and_biomedical_section(playwright, section_name, output_file="mec.json"):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/departments/ebe')

    button_selector = f".sidebar-nav-li:has-text('{section_name}')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    
    section_data = {}
    if section_name == "About":
        page.wait_for_selector('.about')
        section_data["Description of electronics and biomedical department"] = page.locator('.about > .custom-p').text_content().strip()

    # elif section_name == "Vision & Mission":
    #     page.wait_for_selector('.vision-mission')
    #     section_data["Vision of electronics and biomedical department"] = page.locator('.vision .custom-p').text_content().strip()

    #     for item in page.locator('.mission-item').all():
    #         paragraphs = item.locator('.custom-p').all()
    #         if len(paragraphs) >= 2:
    #             mission_number = paragraphs[0].text_content().strip()
    #             mission_text = paragraphs[1].text_content().strip()
    #             section_data[f"Mission {mission_number} of electronics and biomedical department"] = mission_text

    #     for peo in page.locator('h2:text("Program Educational Objectives") ~ .ed-obj .ed-obj-item').all():
    #         if peo.locator('.custom-h3').count() > 0 and peo.locator('.custom-p').count() > 0:
    #             peo_title = peo.locator('.custom-h3').text_content().strip()
    #             peo_desc = peo.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Educational Objective of electronics and biomedical department: {peo_title}"] = peo_desc

    #     for pso in page.locator('h2:text("Program Specific Outcomes") ~ .ed-obj .ed-obj-item').all():
    #         if pso.locator('.custom-h3').count() > 0 and pso.locator('.custom-p').count() > 0:
    #             pso_title = pso.locator('.custom-h3').text_content().strip()
    #             pso_desc = pso.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Specific Outcome of electronics and biomedical department: {pso_title}"] = pso_desc

    #     for po in page.locator('h2:text("Program Outcomes") ~ .ed-obj .ed-obj-item').all():
    #         if po.locator('.custom-h3').count() > 0 and po.locator('.custom-p').count() > 0:
    #             po_title = po.locator('.custom-h3').text_content().strip()
    #             po_desc = po.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Outcome of electronics and biomedical department: {po_title}"] = po_desc

    elif section_name == "Courses Offered":
        page.wait_for_selector('.courses-offered')
        for idx, item in enumerate(page.locator('.course-item').all(), start=1):
            degree = item.locator('.custom-h2').text_content().strip()
            program = f"{degree} {item.locator('.custom-h3.red').first.text_content().strip()}"
            if item.locator('.custom-h3.red').count() > 1:
                specialization = item.locator('.custom-h3.red').nth(1).text_content().strip()
                program += f" with specialization in {specialization}"
            section_data[f"Course {idx} of electronics and biomedical department"] = program

    elif section_name == "HOD":
        page.wait_for_selector('.grid')
        page.wait_for_selector('.photo-item .person-name', state='visible')
        page.wait_for_selector('.photo-item .person-position', state='visible')
        
        section_data["HOD Name of electronics and biomedical department"] = page.locator('.photo-item .person-name').text_content().strip()
        section_data["HOD Position of electronics and biomedical department"] = page.locator('.photo-item .person-position').text_content().strip()
        section_data["HOD Email of electronics and biomedical department"] = page.locator('.bio-contact-item a').get_attribute('href').replace('mailto:', '') if page.locator('.bio-contact-item a').count() > 0 else None
        # section_data["HOD Image URL of electronics and biomedical department"] = page.locator('.photo-item img').get_attribute('src')

    elif section_name == "Faculty":
        page.wait_for_selector('.grid')
        page.wait_for_selector('.photo-item', state='visible')
        page.wait_for_selector('.photo-item .person-name', state='visible')
        page.wait_for_selector('.photo-item .person-position', state='visible')
        
        for idx, member in enumerate(page.locator('.photo-item').all(), start=1):
            section_data[f"Faculty {idx} Name of electronics and biomedical department"] = member.locator('.person-name').text_content().strip()
            section_data[f"Faculty {idx} Position of electronics and biomedical department"] = member.locator('.person-position').text_content().strip()
            # section_data[f"Faculty {idx} Image URL of electronics and biomedical department"] = member.locator('img').get_attribute('src')

    elif section_name == "Facilities":
        page.wait_for_selector('.page-content')
        section_data["Facilities Main Description of electronics and biomedical department"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        
        for idx, facility in enumerate(page.locator('.facility-items > div').all(), start=1):
            name = facility.locator('.custom-h3').text_content().strip()
            desc = facility.locator('.custom-p').text_content().strip()
            section_data[f"Facility {idx} Name of electronics and biomedical department"] = name
            # section_data[f"Facility {idx} Description of electronics and biomedical department"] = desc

    # elif section_name == "Resources":
    #     page.wait_for_selector('.page-content')
    #     section_data["Resources Main Description of electronics and biomedical department"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        
    #     for idx, resource in enumerate(page.locator('.res > div').all(), start=1):
    #         name = resource.locator('.custom-h3').text_content().strip()
    #         desc = resource.locator('.custom-p').text_content().strip()
    #         section_data[f"Resource {idx} Name of electronics and biomedical department"] = name
    #         section_data[f"Resource {idx} Description of electronics and biomedical department"] = desc
            
    #         for link_idx, link in enumerate(resource.locator('.custom-a').all(), start=1):
    #             link_text = link.text_content().strip()
    #             link_url = link.get_attribute('href')
    #             section_data[f"Resource {idx} Link {link_idx} Title of electronics and biomedical department"] = link_text
    #             section_data[f"Resource {idx} Link {link_idx} URL of electronics and biomedical department"] = link_url

    # elif section_name == "Associations":
    #     page.wait_for_selector('.asc')
    #     section_data["Association Name of electronics and biomedical department"] = page.locator('.asc .custom-h3').text_content().strip()
    #     section_data["Association Description of electronics and biomedical department"] = page.locator('.asc .custom-p').text_content().strip()

    # elif section_name == "Achievements":
    #     page.wait_for_selector('.page-content')
    #     section_data["Achievements Main Description of electronics and biomedical department"] = page.locator('.page-content > div > div > .custom-p').text_content().strip()
        
    #     achievement_idx = 1
    #     for item in page.locator('.std-achievements ul > li').all():
    #         achievement_text = item.inner_text().strip()
    #         if item.locator('b').count() > 0:
    #             title = item.locator('b').text_content().strip()
    #             if item.locator('ol').count() > 0:
    #                 for sub_idx, sub_achievement in enumerate(item.locator('ol > li').all(), start=1):
    #                     section_data[f"Achievement {achievement_idx} Title of electronics and biomedical department"] = title
    #                     section_data[f"Achievement {achievement_idx} Sub-achievement {sub_idx} of electronics and biomedical department"] = sub_achievement.text_content().strip()
    #             else:
    #                 content = achievement_text.replace(title, '').strip()
    #                 section_data[f"Achievement {achievement_idx} Title of electronics and biomedical department"] = title
    #                 section_data[f"Achievement {achievement_idx} Description of electronics and biomedical department"] = content
    #         else:
    #             section_data[f"Achievement {achievement_idx} Description of electronics and biomedical department"] = achievement_text
    #         achievement_idx += 1
    # elif section_name == "Associations":
    #     page.wait_for_selector('.asc')
    #     section_data["association_name_of_applied_science"] = page.locator('.asc .custom-h3').text_content().strip()
    #     section_data["association_description_of_applied_science"] = page.locator('.asc .custom-p').text_content().strip()

    # elif section_name == "Recent Projects":
    #     page.wait_for_selector('.page-content')
    #     page.wait_for_selector('.page-content > div > div > .custom-p', state='visible')
    #     section_data["Projects Main Description of electronics and biomedical department"] = page.locator('.page-content > div > div > .custom-p').first.text_content().strip()
    #     page.wait_for_selector('.project-item')
    #     for idx, project in enumerate(page.locator('.project-item').all(), start=1):
    #         title = project.locator('.custom-h3').text_content().strip()
    #         desc = project.locator('.custom-p').text_content().strip()
    #         section_data[f"Project {idx} Title of electronics and biomedical department"] = title
    #         section_data[f"Project {idx} Description of electronics and biomedical department"] = desc
            
    #         for link_idx, link in enumerate(project.locator('a').all(), start=1):
    #             link_text = link.text_content().strip()
    #             link_url = link.get_attribute('href')
    #             section_data[f"Project {idx} Link {link_idx} Title of electronics and biomedical department"] = link_text
    #             section_data[f"Project {idx} Link {link_idx} URL of electronics and biomedical department"] = link_url

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    
    existing_data.update(section_data)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    
    browser.close()

def mechanical_engineering_section(playwright, section_name, output_file="mec.json"):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/departments/me')

    button_selector = f".sidebar-nav-li:has-text('{section_name}')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    
    section_data = {}
    if section_name == "About":
        page.wait_for_selector('.about')
        section_data["Description of mechanical engineering department"] = page.locator('.about > .custom-p').text_content().strip()

    # elif section_name == "Vision & Mission":
    #     page.wait_for_selector('.vision-mission')
    #     section_data["Vision of mechanical engineering department"] = page.locator('.vision .custom-p').text_content().strip()

    #     for item in page.locator('.mission-item').all():
    #         paragraphs = item.locator('.custom-p').all()
    #         if len(paragraphs) >= 2:
    #             mission_number = paragraphs[0].text_content().strip()
    #             mission_text = paragraphs[1].text_content().strip()
    #             section_data[f"Mission {mission_number} of mechanical engineering department"] = mission_text

    #     for peo in page.locator('h2:text("Program Educational Objectives") ~ .ed-obj .ed-obj-item').all():
    #         if peo.locator('.custom-h3').count() > 0 and peo.locator('.custom-p').count() > 0:
    #             peo_title = peo.locator('.custom-h3').text_content().strip()
    #             peo_desc = peo.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Educational Objective of mechanical engineering department: {peo_title}"] = peo_desc

    #     for pso in page.locator('h2:text("Program Specific Outcomes") ~ .ed-obj .ed-obj-item').all():
    #         if pso.locator('.custom-h3').count() > 0 and pso.locator('.custom-p').count() > 0:
    #             pso_title = pso.locator('.custom-h3').text_content().strip()
    #             pso_desc = pso.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Specific Outcome of mechanical engineering department: {pso_title}"] = pso_desc

    #     for po in page.locator('h2:text("Program Outcomes") ~ .ed-obj .ed-obj-item').all():
    #         if po.locator('.custom-h3').count() > 0 and po.locator('.custom-p').count() > 0:
    #             po_title = po.locator('.custom-h3').text_content().strip()
    #             po_desc = po.locator('.custom-p').text_content().strip()
    #             section_data[f"Program Outcome of mechanical engineering department: {po_title}"] = po_desc

    elif section_name == "Courses Offered":
        page.wait_for_selector('.courses-offered')
        for idx, item in enumerate(page.locator('.course-item').all(), start=1):
            degree = item.locator('.custom-h2').text_content().strip()
            program = f"{degree} {item.locator('.custom-h3.red').first.text_content().strip()}"
            if item.locator('.custom-h3.red').count() > 1:
                specialization = item.locator('.custom-h3.red').nth(1).text_content().strip()
                program += f" with specialization in {specialization}"
            section_data[f"Course {idx} of mechanical engineering department"] = program

    elif section_name == "HOD":
        page.wait_for_selector('.grid')
        page.wait_for_selector('.photo-item .person-name', state='visible')
        page.wait_for_selector('.photo-item .person-position', state='visible')
        
        section_data["HOD Name of mechanical engineering department"] = page.locator('.photo-item .person-name').text_content().strip()
        section_data["HOD Position of mechanical engineering department"] = page.locator('.photo-item .person-position').text_content().strip()
        section_data["HOD Email of mechanical engineering department"] = page.locator('.bio-contact-item a').get_attribute('href').replace('mailto:', '') if page.locator('.bio-contact-item a').count() > 0 else None
        # section_data["HOD Image URL of mechanical engineering department"] = page.locator('.photo-item img').get_attribute('src')

    elif section_name == "Faculty":
        page.wait_for_selector('.grid')
        page.wait_for_selector('.photo-item', state='visible')
        page.wait_for_selector('.photo-item .person-name', state='visible')
        page.wait_for_selector('.photo-item .person-position', state='visible')
        
        for idx, member in enumerate(page.locator('.photo-item').all(), start=1):
            section_data[f"Faculty {idx} Name of mechanical engineering department"] = member.locator('.person-name').text_content().strip()
            section_data[f"Faculty {idx} Position of mechanical engineering department"] = member.locator('.person-position').text_content().strip()
            # section_data[f"Faculty {idx} Image URL of mechanical engineering department"] = member.locator('img').get_attribute('src')

    elif section_name == "Facilities":
        page.wait_for_selector('.page-content')
        section_data["Facilities Main Description of mechanical engineering department"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        
        for idx, facility in enumerate(page.locator('.facility-items > div').all(), start=1):
            name = facility.locator('.custom-h3').text_content().strip()
            desc = facility.locator('.custom-p').text_content().strip()
            section_data[f"Facility {idx} Name of mechanical engineering department"] = name
            # section_data[f"Facility {idx} Description of mechanical engineering department"] = desc

    # elif section_name == "Resources":
    #     page.wait_for_selector('.page-content')
    #     section_data["Resources Main Description of mechanical engineering department"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        
    #     for idx, resource in enumerate(page.locator('.res > div').all(), start=1):
    #         name = resource.locator('.custom-h3').text_content().strip()
    #         desc = resource.locator('.custom-p').text_content().strip()
    #         section_data[f"Resource {idx} Name of mechanical engineering department"] = name
    #         section_data[f"Resource {idx} Description of mechanical engineering department"] = desc
            
    #         for link_idx, link in enumerate(resource.locator('.custom-a').all(), start=1):
    #             link_text = link.text_content().strip()
    #             link_url = link.get_attribute('href')
    #             section_data[f"Resource {idx} Link {link_idx} Title of mechanical engineering department"] = link_text
    #             section_data[f"Resource {idx} Link {link_idx} URL of mechanical engineering department"] = link_url

    # elif section_name == "Associations":
    #     page.wait_for_selector('.asc')
    #     section_data["Association Name of mechanical engineering department"] = page.locator('.asc .custom-h3').text_content().strip()
    #     section_data["Association Description of mechanical engineering department"] = page.locator('.asc .custom-p').text_content().strip()

    # elif section_name == "Achievements":
    #     page.wait_for_selector('.page-content')
    #     section_data["Achievements Main Description of mechanical engineering department"] = page.locator('.page-content > div > div > .custom-p').text_content().strip()
        
    #     achievement_idx = 1
    #     for item in page.locator('.std-achievements ul > li').all():
    #         achievement_text = item.inner_text().strip()
    #         if item.locator('b').count() > 0:
    #             title = item.locator('b').text_content().strip()
    #             if item.locator('ol').count() > 0:
    #                 for sub_idx, sub_achievement in enumerate(item.locator('ol > li').all(), start=1):
    #                     section_data[f"Achievement {achievement_idx} Title of mechanical engineering department"] = title
    #                     section_data[f"Achievement {achievement_idx} Sub-achievement {sub_idx} of mechanical engineering department"] = sub_achievement.text_content().strip()
    #             else:
    #                 content = achievement_text.replace(title, '').strip()
    #                 section_data[f"Achievement {achievement_idx} Title of mechanical engineering department"] = title
    #                 section_data[f"Achievement {achievement_idx} Description of mechanical engineering department"] = content
    #         else:
    #             section_data[f"Achievement {achievement_idx} Description of mechanical engineering department"] = achievement_text
    #         achievement_idx += 1
    # elif section_name == "Associations":
    #     page.wait_for_selector('.asc')
    #     section_data["association_name_of_applied_science"] = page.locator('.asc .custom-h3').text_content().strip()
    #     section_data["association_description_of_applied_science"] = page.locator('.asc .custom-p').text_content().strip()

    # elif section_name == "Recent Projects":
    #     page.wait_for_selector('.page-content')
    #     section_data["Projects Main Description of mechanical engineering department"] = page.locator('.page-content > div > div > .custom-p').first.text_content().strip()
        
    #     for idx, project in enumerate(page.locator('.project-item').all(), start=1):
    #         title = project.locator('.custom-h3').text_content().strip()
    #         desc = project.locator('.custom-p').text_content().strip()
    #         section_data[f"Project {idx} Title of mechanical engineering department"] = title
    #         section_data[f"Project {idx} Description of mechanical engineering department"] = desc
            
    #         for link_idx, link in enumerate(project.locator('a').all(), start=1):
    #             link_text = link.text_content().strip()
    #             link_url = link.get_attribute('href')
    #             section_data[f"Project {idx} Link {link_idx} Title of mechanical engineering department"] = link_text
    #             section_data[f"Project {idx} Link {link_idx} URL of mechanical engineering department"] = link_url

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    
    existing_data.update(section_data)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    
    browser.close()

def applied_science_section(playwright, section_name, output_file="mec.json"):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/departments/as')

    button_selector = f".sidebar-nav-li:has-text('{section_name}')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    
    section_data = {}
    if section_name == "About":
        page.wait_for_selector('.about')
        section_data["description_of_applied_science"] = page.locator('.about > .custom-p').text_content().strip()

    # elif section_name == "Vision & Mission":
    #     page.wait_for_selector('.vision-mission')
    #     section_data["vision_of_applied_science"] = page.locator('.vision .custom-p').text_content().strip()

    #     # Process mission items
    #     for idx, item in enumerate(page.locator('.mission-item').all(), start=1):
    #         paragraphs = item.locator('.custom-p').all()
    #         if len(paragraphs) >= 2:
    #             mission_number = paragraphs[0].text_content().strip()
    #             mission_text = paragraphs[1].text_content().strip()
    #             section_data[f"mission_{idx}_of_applied_science"] = mission_text

    #     # Process PEOs
    #     for idx, peo in enumerate(page.locator('h2:text("Program Educational Objectives") ~ .ed-obj .ed-obj-item').all(), start=1):
    #         if peo.locator('.custom-h3').count() > 0 and peo.locator('.custom-p').count() > 0:
    #             title = peo.locator('.custom-h3').text_content().strip()
    #             desc = peo.locator('.custom-p').text_content().strip()
    #             section_data[f"peo_{idx}_title_of_applied_science"] = title
    #             section_data[f"peo_{idx}_description_of_applied_science"] = desc

    #     # Process PSOs
    #     for idx, pso in enumerate(page.locator('h2:text("Program Specific Outcomes") ~ .ed-obj .ed-obj-item').all(), start=1):
    #         if pso.locator('.custom-h3').count() > 0 and pso.locator('.custom-p').count() > 0:
    #             title = pso.locator('.custom-h3').text_content().strip()
    #             desc = pso.locator('.custom-p').text_content().strip()
    #             section_data[f"pso_{idx}_title_of_applied_science"] = title
    #             section_data[f"pso_{idx}_description_of_applied_science"] = desc

    #     # Process POs
    #     for idx, po in enumerate(page.locator('h2:text("Program Outcomes") ~ .ed-obj .ed-obj-item').all(), start=1):
    #         if po.locator('.custom-h3').count() > 0 and po.locator('.custom-p').count() > 0:
    #             title = po.locator('.custom-h3').text_content().strip()
    #             desc = po.locator('.custom-p').text_content().strip()
    #             section_data[f"po_{idx}_title_of_applied_science"] = title
    #             section_data[f"po_{idx}_description_of_applied_science"] = desc

    # elif section_name == "Courses Offered":
    #     page.wait_for_selector('.courses-offered')
    #     for idx, item in enumerate(page.locator('.course-item').all(), start=1):
    #         degree = item.locator('.custom-h2').text_content().strip()
    #         program = f"{degree} {item.locator('.custom-h3.red').first.text_content().strip()}"
            
    #         section_data[f"course_{idx}_degree_of_applied_science"] = degree
    #         if item.locator('.custom-h3.red').count() > 1:
    #             specialization = item.locator('.custom-h3.red').nth(1).text_content().strip()
    #             program += f" with specialization in {specialization}"
    #             section_data[f"course_{idx}_specialization_of_applied_science"] = specialization
    #         section_data[f"course_{idx}_program_of_applied_science"] = program

    elif section_name == "HOD":
        page.wait_for_selector('.grid')
        page.wait_for_selector('.photo-item .person-name', state='visible')
        page.wait_for_selector('.photo-item .person-position', state='visible')
        
        section_data["hod_name_of_applied_science"] = page.locator('.photo-item .person-name').text_content().strip()
        section_data["hod_position_of_applied_science"] = page.locator('.photo-item .person-position').text_content().strip()
        section_data["hod_email_of_applied_science"] = page.locator('.bio-contact-item a').get_attribute('href').replace('mailto:', '') if page.locator('.bio-contact-item a').count() > 0 else None
        # section_data["hod_image_url_of_applied_science"] = page.locator('.photo-item img').get_attribute('src')

    elif section_name == "Faculty":
        page.wait_for_selector('.grid')
        page.wait_for_selector('.photo-item', state='visible')
        page.wait_for_selector('.photo-item .person-name', state='visible')
        page.wait_for_selector('.photo-item .person-position', state='visible')
        
        for idx, member in enumerate(page.locator('.photo-item').all(), start=1):
            section_data[f"faculty_{idx}_name_of_applied_science"] = member.locator('.person-name').text_content().strip()
            section_data[f"faculty_{idx}_position_of_applied_science"] = member.locator('.person-position').text_content().strip()
            # section_data[f"faculty_{idx}_image_url_of_applied_science"] = member.locator('img').get_attribute('src')

    # elif section_name == "Facilities":
    #     page.wait_for_selector('.page-content')
    #     # section_data["facilities_main_description_of_applied_science"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        
    #     for idx, facility in enumerate(page.locator('.facility-items > div').all(), start=1):
    #         name = facility.locator('.custom-h3').text_content().strip()
    #         desc = facility.locator('.custom-p').text_content().strip()
    #         section_data[f"facility_{idx}_name_of_applied_science"] = name
            # section_data[f"facility_{idx}_description_of_applied_science"] = desc

    # elif section_name == "Resources":
    #     page.wait_for_selector('.page-content')
    #     section_data["resources_main_description_of_applied_science"] = page.locator('.page-content > div > .custom-p').first.text_content().strip()
        
    #     for idx, resource in enumerate(page.locator('.res > div').all(), start=1):
    #         name = resource.locator('.custom-h3').text_content().strip()
    #         desc = resource.locator('.custom-p').text_content().strip()
    #         section_data[f"resource_{idx}_name_of_applied_science"] = name
    #         section_data[f"resource_{idx}_description_of_applied_science"] = desc
            
    #         for link_idx, link in enumerate(resource.locator('.custom-a').all(), start=1):
    #             section_data[f"resource_{idx}_link_{link_idx}_title_of_applied_science"] = link.text_content().strip()
    #             section_data[f"resource_{idx}_link_{link_idx}_url_of_applied_science"] = link.get_attribute('href')

    # elif section_name == "Associations":
    #     page.wait_for_selector('.asc')
    #     section_data["association_name_of_applied_science"] = page.locator('.asc .custom-h3').text_content().strip()
    #     section_data["association_description_of_applied_science"] = page.locator('.asc .custom-p').text_content().strip()

    # elif section_name == "Achievements":
    #     page.wait_for_selector('.page-content')
    #     section_data["achievements_main_description_of_applied_science"] = page.locator('.page-content > div > div > .custom-p').text_content().strip()
        
    #     achievement_idx = 1
    #     for item in page.locator('.std-achievements ul > li').all():
    #         if item.locator('b').count() > 0:
    #             title = item.locator('b').text_content().strip()
    #             if item.locator('ol').count() > 0:
    #                 for sub_idx, sub_achievement in enumerate(item.locator('ol > li').all(), start=1):
    #                     section_data[f"achievement_{achievement_idx}_title_of_applied_science"] = title
    #                     section_data[f"achievement_{achievement_idx}_sub_achievement_{sub_idx}_of_applied_science"] = sub_achievement.text_content().strip()
    #             else:
    #                 content = item.inner_text().strip().replace(title, '').strip()
    #                 section_data[f"achievement_{achievement_idx}_title_of_applied_science"] = title
    #                 section_data[f"achievement_{achievement_idx}_description_of_applied_science"] = content
    #         else:
    #             section_data[f"achievement_{achievement_idx}_description_of_applied_science"] = item.inner_text().strip()
    #         achievement_idx += 1

    # elif section_name == "Recent Projects":
    #     page.wait_for_selector('.page-content')
    #     section_data["projects_main_description_of_applied_science"] = page.locator('.page-content > div > div > .custom-p').first.text_content().strip()
        
    #     for idx, project in enumerate(page.locator('.project-item').all(), start=1):
    #         title = project.locator('.custom-h3').text_content().strip()
    #         desc = project.locator('.custom-p').text_content().strip()
    #         section_data[f"project_{idx}_title_of_applied_science"] = title
    #         section_data[f"project_{idx}_description_of_applied_science"] = desc
            
    #         for link_idx, link in enumerate(project.locator('a').all(), start=1):
    #             section_data[f"project_{idx}_link_{link_idx}_text_of_applied_science"] = link.text_content().strip()
    #             section_data[f"project_{idx}_link_{link_idx}_url_of_applied_science"] = link.get_attribute('href')

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    
    existing_data.update(section_data)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    
    browser.close()

def placements_section(playwright, section_name, output_file="mec.json"):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.mec.ac.in/placements')

    button_selector = f".sidebar-nav-li:has-text('{section_name}')"
    page.wait_for_selector(button_selector)
    page.click(button_selector)
    
    section_data = {}
    if section_name == "Activities":
        page.wait_for_selector('.activities')
        for idx, item in enumerate(page.locator('.activities ul li').all(), start=1):
            section_data[f"activity_{idx}_of_placements_section"] = item.text_content().strip()

    # elif section_name == "Placement Statistics":
    #     page.wait_for_selector('.placement-stats')
    #     section_data["statistics_description_of_placements_section"] = page.locator('.special_p').text_content().strip()
        
    #     # Store available years
    #     years = [year.get_attribute('value') for year in page.locator('.select option').all()]
    #     for idx, year in enumerate(years, start=1):
    #         section_data[f"available_year_{idx}_of_placements_section"] = year

    #     # Process company data
    #     for idx, row in enumerate(page.locator('.Table tbody tr').all(), start=1):
    #         style = row.get_attribute('style')
    #         if style and 'background-color' in style:
    #             # Total row
    #             section_data["total_placements_cse_of_placements_section"] = row.locator('td').nth(1).text_content().strip()
    #             section_data["total_placements_ebe_of_placements_section"] = row.locator('td').nth(2).text_content().strip()
    #             section_data["total_placements_ece_of_placements_section"] = row.locator('td').nth(3).text_content().strip()
    #             section_data["total_placements_eee_of_placements_section"] = row.locator('td').nth(4).text_content().strip()
    #             section_data["total_placements_all_departments_of_placements_section"] = row.locator('td').nth(5).text_content().strip()
    #             continue

    #         company_name = row.locator('.company_name').text_content().strip()
    #         # section_data[f"company_{idx}_name_of_placements_section"] = company_name
    #         # section_data[f"company_{idx}_logo_url_of_placements_section"] = row.locator('.logoImage img').get_attribute('src')
    #         section_data[f"{company_name}_placements_cse_of_placements_section"] = row.locator('td').nth(1).text_content().strip()
    #         section_data[f"{company_name}_placements_ebe_of_placements_section"] = row.locator('td').nth(2).text_content().strip()
    #         section_data[f"{company_name}_placements_ece_of_placements_section"] = row.locator('td').nth(3).text_content().strip()
    #         section_data[f"{company_name}_placements_eee_of_placements_section"] = row.locator('td').nth(4).text_content().strip()
    #         section_data[f"{company_name}_placements_total_of_placements_section"] = row.locator('td').nth(5).text_content().strip()

    elif section_name == "Placement Statistics":
        page.wait_for_selector('.placement-stats')
        section_data["statistics_description_of_placements_section"] = page.locator('.special_p').text_content().strip()
        
        # Store available years
        years = [year.get_attribute('value') for year in page.locator('.select option').all()]
        
        # Process company data
        for row in page.locator('.Table tbody tr').all():
            style = row.get_attribute('style')
            if style and 'background-color' in style:
                # Store only the total row data
                section_data["total_placements_cse_of_placements_section"] = row.locator('td').nth(1).text_content().strip()
                section_data["total_placements_ebe_of_placements_section"] = row.locator('td').nth(2).text_content().strip()
                section_data["total_placements_ece_of_placements_section"] = row.locator('td').nth(3).text_content().strip()
                section_data["total_placements_eee_of_placements_section"] = row.locator('td').nth(4).text_content().strip()
                section_data["total_placements_all_departments_of_placements_section"] = row.locator('td').nth(5).text_content().strip()
                continue

            # Process but don't store individual company data
            company_name = row.locator('.company_name').text_content().strip()
            _ = row.locator('td').nth(1).text_content().strip()
            _ = row.locator('td').nth(2).text_content().strip()
            _ = row.locator('td').nth(3).text_content().strip()
            _ = row.locator('td').nth(4).text_content().strip()
            _ = row.locator('td').nth(5).text_content().strip()



    # elif section_name == "Brochure":
    #     page.wait_for_selector('.brochure')
    #     page.wait_for_selector('.brochure-grid a')
    #     for idx, brochure in enumerate(page.locator('.brochure-grid a').all(), start=1):
    #         year = brochure.locator('img').get_attribute('alt')
    #         # section_data[f"brochure_{idx}_year_of_placements_section"] = year
    #         section_data[f"Brochure {year}_pdf_url_of_placements_section"] = brochure.get_attribute('href')
    #         section_data[f"{year}_thumbnail_url_of_placements_section"] = brochure.locator('img').get_attribute('src')

    elif section_name == "Student Verification":
        page.wait_for_selector('.student-verification')
        section_data["verification_description_of_placements_section"] = page.locator('.student-verification > .custom-p').first.text_content().strip()
        
        for idx, item in enumerate(page.locator('.student-verification ul li').all(), start=1):
            section_data[f"verification_requirement_{idx}_of_placements_section"] = item.text_content().strip()
        
        section_data["verification_form_url_of_placements_section"] = page.locator('.student-verification .custom-h3').get_attribute('href')

    elif section_name == "Contact Details":
        page.wait_for_selector('.contact-details')

        # Collect all address lines first
        address_lines = []
        for item in page.locator('.contact-details > ul li').all():
            if not (item.get_attribute('class') and 'flex' in item.get_attribute('class')):
                address_lines.append(item.text_content().strip())

        # Store the combined address as a single line
        if address_lines:
            section_data["contact_address_of_placements_section"] = ", ".join(address_lines)

        # Process phone and email separately
        contact_idx = 1
        for item in page.locator('.contact-details > ul li').all():
            if item.get_attribute('class') and 'flex' in item.get_attribute('class'):
                contact_type = 'Phone' if 'Phone' in item.locator('img').get_attribute('alt') else 'Email'
                section_data[f"contact_{contact_type.lower()}_{contact_idx}_of_placements_section"] = item.text_content().strip().replace('\xa0', ' ').strip()
                contact_idx += 1

        # Process placement committee details
        page.wait_for_selector('.placement-comittee-grid-left')
        grid_left = page.locator('.placement-comittee-grid-left').all()
        page.wait_for_selector('.placement-comittee-grid-right')
        grid_right = page.locator('.placement-comittee-grid-right').all()
        
        for idx, (left, right) in enumerate(zip(grid_left, grid_right), start=1):
            position = left.locator('h3').text_content().strip()
            section_data[f"committee_position_{idx}_of_placements_section"] = position
            
            detail_idx = 1
            for para in right.locator('.custom-p').all():
                for line in para.text_content().strip().split('\n'):
                    line = line.strip()
                    if line:
                        if '@' in line:
                            section_data[f"committee_position_{idx}_email_{detail_idx}_of_placements_section"] = line
                        elif '+91' in line:
                            section_data[f"committee_position_{idx}_phone_{detail_idx}_of_placements_section"] = line
                        else:
                            section_data[f"committee_position_{idx}_name_{detail_idx}_of_placements_section"] = line
                        detail_idx += 1

    elif section_name == "Training Cell":
        page.wait_for_selector('.page-content')
        title = page.locator('h2:has-text("Training Cell")').text_content().strip()
        description = page.locator('h2:has-text("Training Cell") + p.custom-p').text_content().strip()
        section_data["training_cell_description_of_placements_section"] = description

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    
    existing_data.update(section_data)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    
    browser.close()
with sync_playwright() as playwright:
    scrape_principal_details(playwright)
    scrape_about_section(playwright)
    scrape_board_of_governors_section(playwright)
    scrape_administrative_staff_section(playwright)
    scrape_academic_council_section(playwright)
    scrape_pta_section(playwright)
    scrape_senate_section(playwright)    
    scrape_admission_details_section(playwright)
    scrape_facilities(playwright)
    scrape_iqac_section(playwright)
    scrape_grievance_cell(playwright)
    scrape_anti_ragging_committee(playwright)
    scrape_anti_ragging_squad(playwright)
    scrape_safety_manual(playwright)
    comp_sci_section(playwright, "About", output_file="mec.json")
    comp_sci_section(playwright, "Courses Offered", output_file="mec.json")
    comp_sci_section(playwright, "HOD", output_file="mec.json")
    comp_sci_section(playwright, "Faculty", output_file="mec.json")
    comp_sci_section(playwright, "Facilities", output_file="mec.json")

    electronics_and_communication_section(playwright, "About", output_file="mec.json")
    electronics_and_communication_section(playwright, "Courses Offered", output_file="mec.json")
    electronics_and_communication_section(playwright, "HOD", output_file="mec.json")
    electronics_and_communication_section(playwright, "Faculty", output_file="mec.json")
    electronics_and_communication_section(playwright, "Facilities", output_file="mec.json")

    electrical_and_electronics_section(playwright, "About", output_file="mec.json")
    electrical_and_electronics_section(playwright, "Courses Offered", output_file="mec.json")
    electrical_and_electronics_section(playwright, "HOD", output_file="mec.json")
    electrical_and_electronics_section(playwright, "Faculty", output_file="mec.json")
    electrical_and_electronics_section(playwright, "Facilities", output_file="mec.json")

    electronics_and_biomedical_section(playwright, "About", output_file="mec.json")
    electronics_and_biomedical_section(playwright, "Courses Offered", output_file="mec.json")
    electronics_and_biomedical_section(playwright, "HOD", output_file="mec.json")
    electronics_and_biomedical_section(playwright, "Faculty", output_file="mec.json")
    electronics_and_biomedical_section(playwright, "Facilities", output_file="mec.json")

    mechanical_engineering_section(playwright, "About", output_file="mec.json")
    mechanical_engineering_section(playwright, "Courses Offered", output_file="mec.json")
    mechanical_engineering_section(playwright, "HOD", output_file="mec.json")
    mechanical_engineering_section(playwright, "Faculty", output_file="mec.json")
    mechanical_engineering_section(playwright, "Facilities", output_file="mec.json")

    applied_science_section(playwright, "About", output_file="mec.json")
    applied_science_section(playwright, "HOD", output_file="mec.json")
    applied_science_section(playwright, "Faculty", output_file="mec.json")

    placements_section(playwright, "Activities", output_file="mec.json")
    placements_section(playwright, "Placement Statistics", output_file="mec.json")
    placements_section(playwright, "Student Verification", output_file="mec.json")
    placements_section(playwright, "Contact Details", output_file="mec.json")
    placements_section(playwright, "Training Cell", output_file="mec.json")