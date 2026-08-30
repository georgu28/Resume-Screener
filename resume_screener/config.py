"""
Configuration settings for MDST Resume Screener
"""

SEMANTIC_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'resume_sections': ['experience', 'projects', 'skills'],
    'job_sections': ['responsibilities', 'requirements', 'description']
}

# File Paths
PATHS = {
    'pdfs': 'pdfs/',
    'job_descriptions': {
        'Full Stack Developer': 'pdfs/full-stack.pdf',
        'Front End Developer': 'pdfs/front-end.pdf',
        'Product Manager': 'pdfs/product-manager.pdf',
        'Java Developer': 'pdfs/java.pdf'
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Resume Section Titles
SECTION_TITLES = [
    'employment',
    'education', 
    'experience',
    'projects',
    'skills',
    'coursework',
    'research',
    'achievements',
    'technologies',
    'description',
    'responsibilities',
    'requirements',
    'objective',
    'summary',
    'certifications'
]

# Regex Patterns for Information Extraction
REGEX_PATTERNS = {
    'email': r'(?:(?:[\w-]+(?:\.[\w-]+)*)@(?:(?:[\w-]+\.)*\w[\w-]{0,66})\.(?:[a-z]{2,6}(?:\.[a-z]{2})?))',
    'phone': r'(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
    'gpa': r'(?:GPA|gpa)[:\s]*([1-4](?:\.[0-9]{1,2})?(?:/4\.0?|/4)?)',
    'linkedin': r'(?:linkedin\.com/in/[\w-]+|linkedin\.com/profile/view\?id=[\w-]+)',
    'github': r'(?:github\.com/[\w-]+)',
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
}
