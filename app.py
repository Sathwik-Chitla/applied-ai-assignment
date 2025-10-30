import streamlit as st
import os
import pickle
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import time

# Import backend - let Python handle the errors naturally
try:
    from backend import ReviewerRecommendationSystem
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.error("Make sure backend.py is in the same directory and all packages are installed.")
    st.stop()

st.set_page_config(
    page_title="Reviewer Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .reviewer-card {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: white;
    }
    .rank-badge {
        background-color: #4f46e5;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50%;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- Streamlit session state ----------------------
if 'system' not in st.session_state:
    st.session_state.system = None
if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown('<p class="main-header">📚 Reviewer Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered system to match research papers with the best potential reviewers</p>', unsafe_allow_html=True)

# Preferred dataset locations inside repo
PREFERRED_PATHS = [
    "dataset/dataset/authors",
    "dataset/authors",
    "dataset"
]

# ---------------- Sidebar: show chosen dataset path & load status ----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Auto-detect default path
    detected_path = next((p for p in PREFERRED_PATHS if os.path.exists(p) and os.listdir(p)), None)
    
    if detected_path:
        st.success("✅ Dataset detected")
        st.write("**Dataset path:**")
        st.code(detected_path)
        
        # Auto-start loading if not in session_state
        if st.session_state.system is None:
            with st.spinner("Loading dataset..."):
                system = ReviewerRecommendationSystem(detected_path)
                if system.load_dataset():
                    st.session_state.system = system
                    st.success(f"✅ Loaded {len(system.authors)} authors")
                else:
                    st.error("❌ Failed to load dataset. Check folder structure.")
    else:
        st.warning("⚠️ No dataset found in repo. Expected paths:")
        for p in PREFERRED_PATHS:
            st.write(f"- {p}")
    
    st.divider()
    
    # Method selection - map to actual backend method names
    method_display_map = {
        "TF-IDF + Cosine": "method_tfidf_cosine",
        "Jaccard Similarity": "method_jaccard",
        "Topic Modeling (LDA)": "method_lda",
        "Doc2Vec": "method_doc2vec",
        "BERT Embeddings": "method_sentence_bert",
        "Ensemble": "method_ensemble"
    }
    
    method_display = st.selectbox(
        "Matching Method",
        list(method_display_map.keys()),
        help="Select the similarity computation method"
    )
    
    k = st.slider("Number of Reviewers (k)", 1, 20, 5)
    
    st.divider()
    st.info("💡 *Tip:* Upload a PDF paper to get personalized reviewer recommendations")

# ---------------------- Main app tabs ----------------------
tabs = st.tabs(["📄 Upload Paper", "🔍 Results", "📊 Evaluation"])

with tabs[0]:
    st.subheader("Upload Research Paper")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔍 Find Reviewers", type="primary", use_container_width=True):
                if st.session_state.system is None:
                    st.error("❌ Dataset not loaded. App couldn't find dataset in repo.")
                else:
                    # Save uploaded file temporarily
                    with open("temp_paper.pdf", "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    with st.spinner("Analyzing paper and finding reviewers..."):
                        progress_bar = st.progress(0)
                        
                        # Extract text
                        progress_bar.progress(20)
                        text = st.session_state.system.extract_text_from_pdf("temp_paper.pdf")
                        processed_text = st.session_state.system.preprocess_text(text)
                        
                        progress_bar.progress(50)
                        
                        # Get the actual method name
                        method_name = method_display_map[method_display]
                        
                        try:
                            # Call the method
                            method_func = getattr(st.session_state.system, method_name)
                            results = method_func(processed_text, k)
                            
                            progress_bar.progress(100)
                            
                            st.session_state.results = {
                                'recommendations': results,
                                'method': method_display,
                                'paper_name': uploaded_file.name,
                                'total_authors': len(st.session_state.system.authors)
                            }
                            
                            st.success("✅ Analysis complete! Check the Results tab.")
                            time.sleep(1)
                            
                            # Clean up temp file
                            if os.path.exists("temp_paper.pdf"):
                                os.remove("temp_paper.pdf")
                            
                            st.rerun()
                        
                        except AttributeError:
                            st.error(f"❌ Method '{method_name}' not found in backend")
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            if os.path.exists("temp_paper.pdf"):
                                os.remove("temp_paper.pdf")

with tabs[1]:
    if st.session_state.results is not None:
        results = st.session_state.results
        
        st.subheader(f"Top {k} Recommended Reviewers")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Paper", results['paper_name'])
        with col2:
            st.metric("Method", results['method'])
        with col3:
            st.metric("Total Authors", results['total_authors'])
        
        st.divider()
        
        for rec in results['recommendations']:
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.markdown(f'<div style="text-align: center; font-size: 2rem; font-weight: bold; color: #4f46e5;">#{rec["rank"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center; font-size: 1.5rem; color: #4f46e5;">{rec["similarity"]*100:.1f}%</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"### {rec['author']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Publications:** {rec['papers']}")
                    with col_b:
                        st.write(f"**h-index:** {rec['h_index']}")
                    
                    st.write("**Expertise:**")
                    expertise_html = " ".join([
                        f'<span style="background-color: #dbeafe; color: #1e40af; padding: 0.25rem 0.75rem; border-radius: 1rem; margin-right: 0.5rem; display: inline-block;">{exp}</span>'
                        for exp in rec['expertise']
                    ])
                    st.markdown(expertise_html, unsafe_allow_html=True)
                    
                    try:
                        st.progress(rec['similarity'])
                    except Exception:
                        pass
                
                st.divider()
    else:
        st.info("👆 Upload a paper in the 'Upload Paper' tab to see recommendations")

with tabs[2]:
    st.subheader("Method Evaluation")
    
    if st.session_state.system is not None:
        if st.button("🔄 Run Evaluation"):
            with st.spinner("Evaluating methods..."):
                try:
                    evaluation = st.session_state.system.evaluate_methods()
                    
                    st.subheader("Method Performance")
                    
                    for method_info in evaluation['methods']:
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"### {method_info['name']}")
                            
                            with col2:
                                if method_info['success']:
                                    st.metric("Speed", method_info['speed'])
                                else:
                                    st.error("Failed")
                            
                            with col3:
                                if method_info['success']:
                                    st.success("✅")
                                else:
                                    st.error("❌")
                            
                            if not method_info['success'] and 'error' in method_info:
                                st.caption(f"Error: {method_info['error']}")
                            
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
    else:
        st.warning("⚠️ Load dataset first to run evaluation")
    
    st.divider()
    
    st.markdown("""
    ### 📊 Method Descriptions
    
    - **TF-IDF + Cosine**: Fast keyword-based matching using term frequency-inverse document frequency
    - **Jaccard Similarity**: Set-based similarity using n-gram overlap
    - **Topic Modeling (LDA)**: Identifies latent research topics using Latent Dirichlet Allocation
    - **Doc2Vec**: Semantic document similarity using neural embeddings
    - **BERT Embeddings**: State-of-the-art contextual embeddings (requires sentence-transformers)
    - **Ensemble**: Combines multiple approaches for robust matching
    
    ⚠️ **Note:** BERT method requires `sentence-transformers` package which may not be available in all deployments.
    """)

st.divider()
st.caption("Reviewer Recommendation System | Powered by Machine Learning")
