# MDST Resume Screener

A machine learning-powered resume screening application that classifies resumes into job categories and calculates semantic similarity with job descriptions.

## Features

- **KNN Classification**: Automatically categorizes resumes into job types using K-Nearest Neighbors
- **Semantic Analysis**: Calculates similarity between resumes and job descriptions using sentence transformers
- **PDF Processing**: Extracts and parses text from PDF resumes
- **Interactive Dashboard**: Streamlit web interface for easy use
- **Confidence Scoring**: Provides probability scores for predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/georgu28/MDST-Resume-Screener.git
cd MDST-Resume-Screener
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Web Interface
Run the Streamlit app:
```bash
streamlit run app.py
```

### Command Line Usage

#### Resume Classification
```python
from knn_class import ResumeClassifier

classifier = ResumeClassifier()
category = classifier.predict_pdf("path/to/resume.pdf")
print(f"Predicted category: {category}")
```

#### Semantic Similarity
```python
from semantic import SemanticMatcher

matcher = SemanticMatcher()
matcher.get_resume_content("path/to/resume.pdf")
matcher.get_job_description_content("path/to/job_description.pdf")

similarities = matcher.calculate_similarities()
print(f"Similarity score: {similarities[1].item():.3f}")
```

#### PDF Parsing
```python
from parser import read_pdf, get_details, get_sections

# Extract text
text = read_pdf("resume.pdf")

# Extract personal details
details, cleaned_text = get_details(text)
print(f"Email: {details['email']}")
print(f"Phone: {details['phone']}")

# Extract sections
sections = get_sections(text)
print(f"Experience: {sections['experience']}")
```

## Project Structure

```
MDST-Resume-Screener/
├── app.py                 # Streamlit web interface
├── parser.py              # PDF parsing and text extraction
├── semantic.py            # Semantic similarity analysis
├── knn_class.py          # KNN classification model
├── requirements.txt       # Python dependencies
├── UpdatedResumeDataSet.csv  # Training dataset
├── pdfs/                 # Sample PDFs
│   ├── bryan-resume.pdf
│   ├── john-resume.pdf
│   ├── full-stack.pdf
│   └── ...
└── notebooks/            # Jupyter notebooks for experimentation
    ├── knn_resume.ipynb
    ├── semantics.ipynb
    └── ...
```

## Models and Algorithms

### KNN Classifier
- **Algorithm**: K-Nearest Neighbors with TF-IDF vectorization
- **Features**: Text content vectorized using TfidfVectorizer
- **Categories**: Software Engineer, Data Scientist, Product Manager, etc.
- **Evaluation**: Classification report with precision, recall, F1-score

### Semantic Similarity
- **Model**: SentenceTransformer (all-MiniLM-L6-v2)
- **Method**: Cosine similarity between resume and job description embeddings
- **Sections**: Analyzes experience, skills, projects vs. requirements, responsibilities

## API Reference

### ResumeClassifier Class

#### Methods
- `__init__(dataset_path, n_neighbors)`: Initialize classifier
- `predict_pdf(pdf_path)`: Predict category for PDF resume
- `predict_text(text)`: Predict category for text
- `get_prediction_probabilities(pdf_path)`: Get confidence scores
- `get_categories()`: List all available categories

### SemanticMatcher Class

#### Methods
- `__init__(model_name)`: Initialize with sentence transformer model
- `get_resume_content(resume_path)`: Process resume PDF
- `get_job_description_content(description_path)`: Process job description PDF
- `calculate_similarities()`: Compute similarity scores
- `batch_compare_resumes(resume_paths, job_paths)`: Batch processing

### Parser Functions

#### Functions
- `read_pdf(name)`: Extract text from PDF
- `get_email(text)`, `get_phone(text)`, `get_gpa(text)`: Extract specific info
- `get_details(text)`: Extract all personal details
- `get_sections(text)`: Parse into resume sections

## Dataset

The project uses the UpdatedResumeDataSet.csv which contains:
- **Resume**: Text content of resumes
- **Category**: Job category labels
- **Size**: 1000+ resume samples across multiple categories

## Performance

- **KNN Accuracy**: ~85% on test set
- **Semantic Similarity**: Correlation with human judgments: ~0.78
- **Processing Speed**: ~2-3 seconds per resume

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MDST (Michigan Data Science Team) for project inspiration
- Sentence Transformers library for semantic analysis
- scikit-learn for machine learning tools
- Streamlit for the web interface
