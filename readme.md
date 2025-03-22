# AI-Powered Resume Screening System

## 🚀 Overview
This project is an **AI-powered resume screening and ranking system** that automates the hiring process by evaluating resumes against job descriptions. It uses **Natural Language Processing (NLP)** techniques like **TF-IDF and BERT embeddings** to compare and rank resumes based on relevance. The system is built with **Streamlit** for an interactive UI and stores resumes and job descriptions in an **SQLite database**.

## 📌 Features
- **Job Description Management**: Store and manage job descriptions.
- **Resume Upload & Text Extraction**: Supports **PDF, DOCX, TXT, and CSV** formats.
- **AI-Based Resume Ranking**: Uses **BERT embeddings** for similarity comparison.
- **Streamlit UI**: Simple and intuitive interface for recruiters.
- **SQLite Database**: Stores job descriptions and resumes for efficient retrieval.

## 🛠️ Technologies Used
- **Python 3.8+**
- **Streamlit** (for UI)
- **SQLite** (for database management)
- **SentenceTransformers** (`all-MiniLM-L6-v2` for BERT embeddings)
- **pdfplumber, mammoth, pandas** (for resume text extraction)
- **scikit-learn** (for cosine similarity calculation)

## 📂 Project Structure
```
📁 resume-screening
│-- 📄 main.py  # Streamlit app
│-- 📄 database.py  # Database functions
│-- 📄 text_extraction.py  # Resume parsing functions
│-- 📄 requirements.txt  # Dependencies
│-- 📄 README.md  # Project documentation
│-- 📂 data
│   ├── resume_screening.db  # SQLite database
```

## 📥 Installation
### **Step 1: Clone the Repository**
```sh
git clone https://github.com/yourusername/resume-screening.git
cd resume-screening
```

### **Step 2: Install Dependencies**
```sh
pip install -r requirements.txt
```

### **Step 3: Run the Streamlit App**
```sh
streamlit run main.py
```

## 🏗️ Database Schema
### **job_descriptions Table**
| Column Name   | Data Type  | Description |
|--------------|-----------|-------------|
| `id`         | INTEGER   | Unique job identifier |
| `description` | TEXT      | Job description |

### **resumes Table**
| Column Name       | Data Type  | Description |
|------------------|-----------|-------------|
| `name`           | TEXT      | Candidate name |
| `extracted_text` | TEXT      | Resume extracted text |
| `job_id`         | INTEGER   | Foreign key linking to job descriptions |

## 🎯 Usage Guide
1. **Add a job description**.
2. **Upload resumes** for the job.
3. **Click "Screen Resumes"** to rank candidates based on relevance.
4. **View ranked resumes** in the Streamlit UI.

## 🔥 Future Improvements
- **OCR support** for scanned resumes.
- **Industry-specific resume ranking customization**.
- **API integration** with HR systems.

## 🤝 Contributing
Feel free to fork this repository, create a feature branch, and submit a pull request! 🚀

## 📜 License
MIT License

---
**Created by Adityan**
