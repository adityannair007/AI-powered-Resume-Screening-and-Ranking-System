import os
import sqlite3
import pdfplumber
import mammoth
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Database setup
DB_NAME = "resume_screening.db"

def create_database():
    """Creates database tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS job_descriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT,
        phone TEXT,
        skills TEXT,
        extracted_text TEXT NOT NULL,
        job_id INTEGER,
        FOREIGN KEY (job_id) REFERENCES job_descriptions (id)
    );
    ''')

    conn.commit()
    conn.close()

create_database()  # Ensure the database is initialized

def insert_job_description(title, description):
    """Insert a job description into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO job_descriptions (title, description) VALUES (?, ?)", (title, description))
    conn.commit()
    conn.close()

def insert_resume(name, email, phone, skills, extracted_text, job_id):
    """Insert resume details into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO resumes (name, email, phone, skills, extracted_text, job_id) VALUES (?, ?, ?, ?, ?, ?)",
                   (name, email, phone, skills, extracted_text, job_id))
    conn.commit()
    conn.close()

def get_resumes_for_job(job_id):
    """Fetch resumes for a specific job description."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, extracted_text FROM resumes WHERE job_id = ?", (job_id,))
    resumes = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in resumes}  # Returns a dict {name: text}

def extract_text_from_file(file):
    """Extract text from various file formats."""
    if file.name.endswith(".txt"):
        return file.getvalue().decode("utf-8")
    elif file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        return mammoth.extract_raw_text(file).value
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return " ".join(df.astype(str).values.flatten())
    return ""

def calculate_relevance(job_description, resumes):
    """Compute similarity scores between the job description and resumes using BERT."""
    if not resumes:
        return {}

    texts = [job_description] + list(resumes.values())
    embeddings = model.encode(texts, convert_to_tensor=True).to("cpu")

    job_embedding = embeddings[0].unsqueeze(0).numpy()
    resume_embeddings = embeddings[1:].numpy()
    
    scores = cosine_similarity(job_embedding, resume_embeddings).flatten()
    return dict(zip(resumes.keys(), scores))

# Streamlit UI
st.title("AI-Powered Resume Screening System")

# Job Description Input
st.header("Add Job Description")
job_title = st.text_input("Job Title")
job_description = st.text_area("Enter the Job Description")
if st.button("Save Job Description"):
    if job_title and job_description:
        insert_job_description(job_title, job_description)
        st.success("Job description saved successfully!")

# Select Job Description
st.header("Select Job Description for Screening")
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT id, title FROM job_descriptions")
jobs = cursor.fetchall()
conn.close()

job_options = {str(job[0]): job[1] for job in jobs}
selected_job_id = st.selectbox("Select a Job", options=list(job_options.keys()), format_func=lambda x: job_options[x])

# Resume Upload Section
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload Resumes (TXT, PDF, DOCX, CSV)", type=["txt", "pdf", "docx", "csv"], accept_multiple_files=True)

if st.button("Save Resumes"):
    if uploaded_files and selected_job_id:
        for file in uploaded_files:
            extracted_text = extract_text_from_file(file)
            if extracted_text.strip():
                insert_resume(file.name, None, None, None, extracted_text, int(selected_job_id))
        st.success("Resumes saved successfully!")

# Screening Resumes
if st.button("Screen Resumes"):
    if not selected_job_id:
        st.error("Please select a job description.")
    else:
        # Fetch job description
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT description FROM job_descriptions WHERE id = ?", (selected_job_id,))
        job_desc = cursor.fetchone()
        conn.close()

        if not job_desc:
            st.error("Job description not found.")
        else:
            job_description_text = job_desc[0]
            resumes = get_resumes_for_job(int(selected_job_id))

            if not resumes:
                st.error("No resumes found for this job.")
            else:
                # Calculate similarity scores
                scores = calculate_relevance(job_description_text, resumes)

                # Display results
                df = pd.DataFrame(list(scores.items()), columns=["Resume", "Relevance Score"])
                df = df.sort_values(by="Relevance Score", ascending=False)
                st.write("### Ranked Resumes:")
                st.dataframe(df)
