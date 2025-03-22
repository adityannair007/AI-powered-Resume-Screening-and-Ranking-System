import pdfplumber
import mammoth as mm
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def extract_text_from_file(file):
    """Extract text from various file formats."""
    if file.name.endswith(".txt"):
        return file.getvalue().decode("utf-8")
    elif file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = mm.extract_raw_text(file)
        return doc.value
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return " ".join(df.astype(str).values.flatten())
    return ""

def calculate_relevance(job_description, resumes):
    """Compute similarity scores between the job description and resumes."""
    all_texts = [job_description] + list(resumes.values())

    embeddings = model.encode(all_texts, convert_to_tensor=True)

    job_embeddings = embeddings[0].unsqueeze(0)
    resume_embeddings = embeddings[1:]

    scores = cosine_similarity(job_embeddings, resume_embeddings).flatten()
    return dict(zip(resumes.keys(), scores))

def main():
    st.title("AI-Powered Resume Screening System")
    
  
    job_description = st.text_area("Enter the Job Description:")
    

    uploaded_files = st.file_uploader("Upload Resumes (TXT, PDF, DOCX, CSV)", type=["txt", "pdf", "docx", "csv"], accept_multiple_files=True)
    
    if st.button("Screen Resumes"):
        if not job_description:
            st.error("Please enter a job description.")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one resume file.")
            return
        

        resumes = {file.name: extract_text_from_file(file) for file in uploaded_files if extract_text_from_file(file)}     
        scores = calculate_relevance(job_description, resumes)

        df = pd.DataFrame(list(scores.items()), columns=["Resume", "Relevance Score"])
        df = df.sort_values(by="Relevance Score", ascending=False)
        st.write("### Ranked Resumes:")
        st.dataframe(df)
        
if __name__ == "__main__":
    main()
