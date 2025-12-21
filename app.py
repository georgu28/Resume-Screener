from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import io
import tempfile
import os
from parser import read_pdf
from semantic import SemanticMatcher
from knn_class import ResumeClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MDST Resume Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MDST Resume Screener")
st.markdown("Upload a resume PDF to analyze its content and get job category predictions")
st.divider()

# File uploader
st.subheader("Upload Resume")
file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload a PDF resume file to begin analysis")

if file:
    try:
        file_value = file.getvalue()
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_value)
            tmp_file_path = tmp_file.name

        # Extract text from PDF
        text = read_pdf(tmp_file_path)
        
        if text.strip():
            # Main content area with tabs for better organization
            tab1, tab2, tab3 = st.tabs(["Analysis Results", "Resume Preview", "Extracted Text"])
            
            with tab1:
                # KNN Classification Section
                st.header("Job Category Prediction")
                st.markdown("AI-powered classification using K-Nearest Neighbors algorithm")
                
                with st.spinner("Analyzing resume with KNN classifier..."):
                    try:
                        classifier = ResumeClassifier()
                        predicted_category = classifier.predict_pdf(tmp_file_path)
                        
                        # Display prediction in a prominent way
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.info(f"**Predicted Category:** {predicted_category}")
                        
                        # Get prediction probabilities
                        probabilities = classifier.get_prediction_probabilities(tmp_file_path)
                        
                        st.subheader("Category Confidence Scores")
                        st.markdown("Top predictions with confidence levels")
                        
                        # Display top 5 with better formatting
                        top_probabilities = dict(list(probabilities.items())[:5])
                        
                        # Create columns for better layout
                        for category, prob in top_probabilities.items():
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.markdown(f"**{category}**")
                            with col2:
                                st.progress(prob, text=f"{prob:.1%}")
                            with col3:
                                st.markdown(f"{prob:.3f}")
                        
                    except Exception as e:
                        st.error(f"Error in KNN prediction: {str(e)}")
                        logger.error(f"KNN prediction error: {e}")
                
                st.divider()
                
                # Semantic Similarity Analysis Section
                st.header("Semantic Similarity Analysis")
                st.markdown("Compare resume against job descriptions using semantic analysis")
                
                job_descriptions = {
                    "Full Stack Developer": "pdfs/full-stack.pdf",
                    "Front End Developer": "pdfs/front-end.pdf", 
                    "Product Manager": "pdfs/product-manager.pdf",
                    "Java Developer": "pdfs/java.pdf"
                }
                
                with st.spinner("Calculating semantic similarities..."):
                    try:
                        matcher = SemanticMatcher()
                        
                        # Process resume
                        matcher.get_resume_content(tmp_file_path)
                        
                        similarities = {}
                        for job_title, job_path in job_descriptions.items():
                            if os.path.exists(job_path):
                                # Add job description
                                matcher.get_job_description_content(job_path)
                                
                                # Calculate similarity
                                sim_scores = matcher.calculate_similarities()
                                if sim_scores is not None and len(sim_scores) > 1:
                                    similarities[job_title] = round(sim_scores[1].item(), 3)
                                
                                # Remove job description for next iteration
                                if len(matcher.sentences) > 1:
                                    matcher.sentences.pop()
                        
                        if similarities:
                            st.subheader("Job Match Scores")
                            st.markdown("Similarity scores compared to job descriptions (higher is better)")
                            
                            # Sort by similarity score
                            sorted_similarities = dict(sorted(similarities.items(), 
                                                             key=lambda x: x[1], reverse=True))
                            
                            # Display with better formatting
                            for job_title, score in sorted_similarities.items():
                                col1, col2, col3 = st.columns([3, 2, 1])
                                with col1:
                                    st.markdown(f"**{job_title}**")
                                with col2:
                                    # Normalize score for progress bar (assuming max is around 1.0)
                                    normalized_score = min(score, 1.0)
                                    st.progress(normalized_score, text=f"{score:.3f}")
                                with col3:
                                    score_percent = min(score * 100, 100)
                                    st.metric("Match", f"{score_percent:.1f}%")
                            
                            st.divider()
                            
                            # Recommend best match with better styling
                            best_match = max(similarities, key=similarities.get)
                            best_score = similarities[best_match]
                            
                            st.success(f"**Best Match:** {best_match} | **Score:** {best_score:.3f}")
                        else:
                            st.warning("No similarities could be calculated. Please ensure job description PDFs are available.")
                            
                    except Exception as e:
                        st.error(f"Error in semantic analysis: {str(e)}")
                        logger.error(f"Semantic analysis error: {e}")
            
            with tab2:
                st.header("Resume Preview")
                pdf_viewer(file_value)
            
            with tab3:
                st.header("Extracted Text")
                st.markdown("Raw text extracted from the PDF")
                with st.expander("View full extracted text", expanded=False):
                    st.text_area("Extracted Text", text, height=400, label_visibility="collapsed")
                st.caption(f"Total characters extracted: {len(text)}")
        else:
            st.error("No text could be extracted from the PDF. Please ensure the PDF contains readable text.")
            
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
else:
    st.info("Please upload a PDF resume file to begin analysis.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This resume screener uses:
    
    **KNN Classification**
    - Trained on resume dataset
    - Predicts job categories
    - Shows confidence scores
    
    **Semantic Analysis** 
    - Uses sentence transformers
    - Compares resume to job descriptions
    - Finds best job matches
    
    **Features**
    - PDF text extraction
    - Multi-model analysis
    - Interactive results
    """)
    
    st.header("Supported Categories")
    st.markdown("""
    - Full Stack Developer
    - Front End Developer  
    - Product Manager
    - Java Developer
    - And more...
    """)