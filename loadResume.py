import fitz  # PyMuPDF
import os
from datetime import datetime

def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(stream=pdf_path.read(), filetype="pdf")
    # doc = fitz.open(pdf_path)
    sections = {}
    current_section = None
    candidate_name = None

    # Define possible sections with synonyms
    section_synonyms = {
        'Contact Information': ['Contact Information', 'Contact Details', 'Personal Information'],
        'Summary': ['Summary', 'Professional Summary', 'Profile'],
        'Objective': ['Objective', 'Career Objective', 'Professional Goal'],
        'Experience': ['EXPERIENCE', 'Experience', 'Work Experience', 'Professional Experience'],
        'Education': ['EDUCATION', 'Education', 'Academic Background', 'Qualifications'],
        'Skills': ['Skills', 'Core Competencies', 'Expertise'],
        'Certifications': ['Certifications', 'Certifications & Licenses', 'Credentials'],
        'Projects': ['PROJECTS', 'Projects', 'Project Experience', 'Significant Projects'],
        'Languages': ['Languages', 'Language Skills', 'Language Proficiency'],
        'Awards': ['Awards', 'Honors', 'Achievements'],
        'Volunteer Experience': ['Volunteer Experience', 'Volunteering', 'Community Service'],
        'Publications': ['Publications', 'Research Papers', 'Articles']
    }
    
    # Create a dictionary to map synonyms to standard section names
    synonym_to_section = {synonym: section for section, synonyms in section_synonyms.items() for synonym in synonyms}

    # Iterate through pages
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        
        for block in blocks:
            block_text = block[4].strip()  # Extract the text part of the block
            if not block_text:
                continue

            # Extract the candidate's name (usually the first line or prominent text)
            if not candidate_name:
                candidate_name = extract_candidate_name(block_text)

            # Identify the section based on heading text
            identified = False
            for heading, synonyms in section_synonyms.items():
                if any(block_text.lower().startswith(synonym.lower()) for synonym in synonyms):
                    current_section = heading
                    if current_section not in sections:
                        sections[current_section] = ''
                    identified = True
                    break
            if identified and current_section:
                # Append text to the current section
                sections[current_section] += block_text + '\n'
    
    return candidate_name, sections

def extract_candidate_name(text):
    # Basic name extraction heuristic: assume the name is the first non-empty line
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and not any(word in line for word in ['Experience', 'Education', 'Skills', 'Summary', 'Objective']):
            return line
    return None

def get_last_modified_date(file_path):
    # Get the last modified time of the file
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except FileNotFoundError:
        return "File not found"

def main():
    pdf_path = 'resume.pdf'
    candidate_name, sections = extract_sections_from_pdf(pdf_path)
    last_modified_date = get_last_modified_date(pdf_path)
    
    # Print candidate's name, last modified date, and extracted sections
    if candidate_name:
        print(f"Candidate Name: {candidate_name}")
    else:
        print("Candidate Name: Not found")
    
    print(f"Last Modified Date: {last_modified_date}")
    
    for section, content in sections.items():
        print(f"--- {section} ---")
        print(content)
        print()

if __name__ == "__main__":
    main()
