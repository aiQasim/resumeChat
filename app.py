import streamlit as st
import pdfplumber
import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from loadResume import *

# Streamlit UI setup
st.title("Resume Talks")
st.write("Upload your resume as a PDF (Max size: 1 MB) and ask questions about your Education, Work, Research, or Skills.")

# Set the maximum file size to 1 MB (1,048,576 bytes)
MAX_FILE_SIZE_MB = 1 * 1024 * 1024  # 1 MB in bytes

# File uploader for the resume
uploaded_file = st.file_uploader("Upload your resume PDF", type="pdf")

if uploaded_file is not None:
    # Check the file size
    if uploaded_file.size > MAX_FILE_SIZE_MB:
        st.error("The file size exceeds the maximum limit of 1 MB. Please upload a smaller file.")
    else:
        # Extract resume text from uploaded PDF
        candidate_name, sections = extract_sections_from_pdf(uploaded_file)
        # last_modified_date = get_last_modified_date(uploaded_file)

        # resume_text = extract_text_from_pdf(uploaded_file)
        
        # Split the resume into sections (Education, Work, Research, Skills)
        # sections = split_into_sections(resume_text)
        
        st.write("Resume Sections Extracted:")
        for section, content in sections.items():
            st.subheader(section)
            st.text_area(f"{section} Section", content, height=150)
        
        # Load and split the resume text for each section
        embedded_sections = {}
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        for section, content in sections.items():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_text(content)
            embedded_sections[section] = embeddings.embed_documents(texts)

        # Load a pre-trained question-answering model (DistilBERT fine-tuned on SQuAD)
        #llm = HuggingFaceHub(repo_id="distilbert-base-uncased-distilled-squad", model_kwargs={"temperature": 0})
        llm = HuggingFaceHub(repo_id="distilbert/distilbert-base-uncased-distilled-squad", model_kwargs={"temperature": 0})
        

        # Create the QA chain
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        # Ask a question to the chatbot
        user_question = st.text_input("Ask a question about your resume (e.g., 'What is my work experience?' or 'Tell me about my research'):")

        if user_question:
            # Identify the section most relevant to the user's question
            relevant_section = None
            for section in sections:
                if section.lower() in user_question.lower():
                    relevant_section = section
                    break

            if relevant_section:
                st.write(f"Searching the {relevant_section} section for an answer...")
                result = qa_chain.run(input_documents=embedded_sections[relevant_section], question=user_question)
                st.write("Answer:", result)
            else:
                st.write("Please ask about one of the following sections: Education, Work, Research, Skills.")
