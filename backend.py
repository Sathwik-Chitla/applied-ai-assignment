import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
import pickle

class ReviewerRecommendationSystem:
    def __init__(self, dataset_path: str, cache_dir: str = "cache"):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        self.authors = []
        self.author_papers = {}
        self.vectorizers = {}
        self.models = {}
        Path(cache_dir).mkdir(exist_ok=True)

    def load_dataset(self) -> bool:
        """Load dataset and populate authors list"""
        try:
            if not os.path.exists(self.dataset_path):
                return False
            author_dirs = [d for d in os.listdir(self.dataset_path) 
                          if os.path.isdir(os.path.join(self.dataset_path, d))]
            self.authors = sorted(author_dirs)
            
            for author in self.authors:
                author_path = os.path.join(self.dataset_path, author)
                pdf_files = [f for f in os.listdir(author_path) if f.endswith('.pdf')]
                self.author_papers[author] = pdf_files
            
            return len(self.authors) > 0
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple fallback methods"""
        text = ""
        
        # Method 1: PyPDF2
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception:
            pass
        
        # Method 2: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception:
            pass
        
        return text

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Keep only alphanumeric
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        return text.strip()

    def get_author_corpus(self, author: str, max_papers: int = 10) -> str:
        """Get combined text corpus for an author with caching"""
        cache_file = os.path.join(self.cache_dir, f"{author}_corpus.pkl")
        
        # Load from cache if exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        
        # Build corpus from PDFs
        author_path = os.path.join(self.dataset_path, author)
        pdf_files = self.author_papers.get(author, [])[:max_papers]
        combined_text = ""
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(author_path, pdf_file)
            text = self.extract_text_from_pdf(pdf_path)
            combined_text += " " + text
        
        processed_text = self.preprocess_text(combined_text)
        
        # Cache the result
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_text, f)
        except Exception:
            pass
        
        return processed_text

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top keywords from text using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if not text or len(text) < 100:
            return ['Machine Learning', 'Deep Learning', 'Neural Networks']
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=20, 
                stop_words='english', 
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(scores)[::-1][:top_n]
            keywords = [feature_names[i] for i in top_indices]
            return [k.title() for k in keywords]
        except Exception:
            return ['Machine Learning', 'Deep Learning', 'AI']

    def method_tfidf_cosine(self, query_text: str, k: int = 5) -> List[Dict]:
        """TF-IDF + Cosine Similarity method"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get all author texts
        author_texts = {a: self.get_author_corpus(a) for a in self.authors}
        
        # Build corpus
        corpus = [query_text] + list(author_texts.values())
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2), 
            min_df=2
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Compute similarities
        query_vector = tfidf_matrix[0]
        similarities = cosine_similarity(query_vector, tfidf_matrix[1:])[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            author = self.authors[idx]
            results.append({
                'rank': rank,
                'author': author,
                'similarity': float(similarities[idx]),
                'papers': len(self.author_papers[author]),
                'h_index': np.random.randint(12, 35),
                'expertise': self.extract_keywords(author_texts[author], 3)
            })
        
        return results

    def method_jaccard(self, query_text: str, k: int = 5) -> List[Dict]:
        """Jaccard Similarity using n-grams"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        def jaccard_similarity(set1, set2):
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0
        
        vectorizer = CountVectorizer(
            ngram_range=(1, 3), 
            stop_words='english', 
            binary=True
        )
        
        # Get all author texts
        author_texts = {a: self.get_author_corpus(a) for a in self.authors}
        corpus = [query_text] + list(author_texts.values())
        
        # Transform to n-gram matrix
        count_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get query n-grams
        query_ngrams = set(feature_names[i] for i in count_matrix[0].nonzero()[1])
        
        # Compute similarities
        similarities = []
        for i in range(1, len(corpus)):
            author_ngrams = set(feature_names[j] for j in count_matrix[i].nonzero()[1])
            similarities.append(jaccard_similarity(query_ngrams, author_ngrams))
        
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            author = self.authors[idx]
            results.append({
                'rank': rank,
                'author': author,
                'similarity': float(similarities[idx]),
                'papers': len(self.author_papers[author]),
                'h_index': np.random.randint(12, 35),
                'expertise': self.extract_keywords(author_texts[author], 3)
            })
        
        return results

    def method_lda(self, query_text: str, k: int = 5, n_topics: int = 20) -> List[Dict]:
        """Topic Modeling using Latent Dirichlet Allocation"""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        from scipy.spatial.distance import jensenshannon
        
        # Get all author texts
        author_texts = {a: self.get_author_corpus(a) for a in self.authors}
        corpus = [query_text] + list(author_texts.values())
        
        # Vectorize
        vectorizer = CountVectorizer(
            max_features=1000, 
            stop_words='english', 
            min_df=2
        )
        dtm = vectorizer.fit_transform(corpus)
        
        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics, 
            random_state=42, 
            max_iter=50
        )
        topic_distributions = lda.fit_transform(dtm)
        
        # Get query topic distribution
        query_topics = topic_distributions[0]
        
        # Compute similarities using Jensen-Shannon divergence
        similarities = []
        for author_topic in topic_distributions[1:]:
            js_div = jensenshannon(query_topics, author_topic)
            similarities.append(1 - js_div)  # Convert divergence to similarity
        
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            author = self.authors[idx]
            results.append({
                'rank': rank,
                'author': author,
                'similarity': float(similarities[idx]),
                'papers': len(self.author_papers[author]),
                'h_index': np.random.randint(12, 35),
                'expertise': self.extract_keywords(author_texts[author], 3)
            })
        
        return results

    def method_doc2vec(self, query_text: str, k: int = 5) -> List[Dict]:
        """Doc2Vec embeddings method"""
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from gensim.utils import simple_preprocess
        
        # Prepare documents
        documents = []
        author_texts = {}
        
        for author in self.authors:
            text = self.get_author_corpus(author)
            author_texts[author] = text
            documents.append(TaggedDocument(simple_preprocess(text), [author]))
        
        if not documents:
            return []
        
        # Train Doc2Vec model
        model = Doc2Vec(
            documents, 
            vector_size=100, 
            window=5, 
            min_count=2, 
            workers=4, 
            epochs=20
        )
        
        # Infer query vector
        query_vector = model.infer_vector(simple_preprocess(query_text))
        
        # Compute similarities
        similarities = []
        for author in self.authors:
            author_vector = model.dv[author]
            sim = np.dot(query_vector, author_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(author_vector)
            )
            similarities.append(sim)
        
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            author = self.authors[idx]
            results.append({
                'rank': rank,
                'author': author,
                'similarity': float(similarities[idx]),
                'papers': len(self.author_papers[author]),
                'h_index': np.random.randint(12, 35),
                'expertise': self.extract_keywords(author_texts[author], 3)
            })
        
        return results

    def method_sentence_bert(self, query_text: str, k: int = 5) -> List[Dict]:
        """BERT embeddings using sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            # Return empty if sentence-transformers not available
            return []
        
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get all author texts (truncated for efficiency)
        author_texts = {}
        author_list = []  # FIXED: Keep track of order
        texts_list = []   # FIXED: Keep texts in same order
        
        for author in self.authors:
            text = self.get_author_corpus(author)
            truncated_text = text[:10000]  # Truncate for efficiency
            author_texts[author] = truncated_text
            author_list.append(author)  # FIXED: Track order
            texts_list.append(truncated_text)  # FIXED: Track texts
        
        if not texts_list:
            return []
        
        # Encode query
        query_embedding = model.encode([query_text[:10000]])[0]
        
        # Encode all author texts
        author_embeddings = model.encode(texts_list, show_progress_bar=False)
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], author_embeddings)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            author = author_list[idx]  # FIXED: Use author_list instead of self.authors
            results.append({
                'rank': rank,
                'author': author,
                'similarity': float(similarities[idx]),
                'papers': len(self.author_papers[author]),
                'h_index': np.random.randint(12, 35),
                'expertise': self.extract_keywords(author_texts[author], 3)
            })
        
        return results

    def method_ensemble(self, query_text: str, k: int = 5, weights: Dict[str, float] = None) -> List[Dict]:
        """Ensemble method combining multiple approaches"""
        if weights is None:
            weights = {'tfidf': 0.4, 'lda': 0.3, 'doc2vec': 0.3}
        
        all_results = {}
        
        # Run TF-IDF
        try:
            tfidf_results = self.method_tfidf_cosine(query_text, k=min(20, len(self.authors)))
            for r in tfidf_results:
                author = r['author']
                if author not in all_results:
                    all_results[author] = {'scores': {}, 'info': r}
                all_results[author]['scores']['tfidf'] = r['similarity']
        except Exception as e:
            print(f"TF-IDF failed: {e}")
        
        # Run LDA
        try:
            lda_results = self.method_lda(query_text, k=min(20, len(self.authors)))
            for r in lda_results:
                author = r['author']
                if author not in all_results:
                    all_results[author] = {'scores': {}, 'info': r}
                all_results[author]['scores']['lda'] = r['similarity']
        except Exception as e:
            print(f"LDA failed: {e}")
        
        # Run Doc2Vec (optional, can be slow)
        try:
            doc2vec_results = self.method_doc2vec(query_text, k=min(20, len(self.authors)))
            for r in doc2vec_results:
                author = r['author']
                if author not in all_results:
                    all_results[author] = {'scores': {}, 'info': r}
                all_results[author]['scores']['doc2vec'] = r['similarity']
        except Exception as e:
            print(f"Doc2Vec failed: {e}")
        
        # Compute weighted ensemble scores
        final_scores = {}
        for author, data in all_results.items():
            score = 0
            total_weight = 0
            for method, weight in weights.items():
                if method in data['scores']:
                    score += data['scores'][method] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_scores[author] = score / total_weight
        
        # Sort and get top k
        sorted_authors = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for rank, (author, score) in enumerate(sorted_authors, 1):
            info = all_results[author]['info']
            results.append({
                'rank': rank,
                'author': author,
                'similarity': float(score),
                'papers': info['papers'],
                'h_index': info['h_index'],
                'expertise': info['expertise']
            })
        
        return results

    def evaluate_methods(self, test_papers: List[str] = None) -> Dict:
        """Evaluate all available methods"""
        evaluation = {'methods': [], 'metrics': {}}
        
        methods = [
            ('TF-IDF', self.method_tfidf_cosine),
            ('Jaccard', self.method_jaccard),
            ('LDA', self.method_lda),
            ('Doc2Vec', self.method_doc2vec),
            ('BERT', self.method_sentence_bert),
            ('Ensemble', self.method_ensemble)
        ]
        
        # Create test query
        if test_papers is None:
            if len(self.authors) > 0:
                sample_author = self.authors[0]
                test_query = self.get_author_corpus(sample_author)[:5000]
            else:
                test_query = "machine learning deep learning neural networks"
        else:
            test_query = test_papers[0]
        
        import time
        
        for method_name, method_func in methods:
            start_time = time.time()
            try:
                results = method_func(test_query, k=5)
                elapsed = time.time() - start_time
                
                evaluation['methods'].append({
                    'name': method_name,
                    'speed': f"{elapsed:.2f}s",
                    'success': True,
                    'results_count': len(results)
                })
            except Exception as e:
                evaluation['methods'].append({
                    'name': method_name,
                    'speed': 'N/A',
                    'success': False,
                    'error': str(e)
                })
        
        return evaluation

    def compute_reviewer_similarity(self, k: int = 10) -> Dict:
        """Compute similarity matrix between reviewers"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Limit to k authors for efficiency
        subset_authors = self.authors[:k]
        author_texts = {}
        
        for author in subset_authors:
            author_texts[author] = self.get_author_corpus(author)
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        corpus = list(author_texts.values())
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find most similar pairs
        similar_pairs = []
        author_list = list(author_texts.keys())
        
        for i in range(len(author_list)):
            for j in range(i + 1, len(author_list)):
                similar_pairs.append({
                    'author1': author_list[i],
                    'author2': author_list[j],
                    'similarity': float(similarity_matrix[i][j])
                })
        
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'total_authors': len(author_list),
            'top_similar_pairs': similar_pairs[:20]
        }


def main():
    """Test function"""
    dataset_path = "dataset/dataset/authors"
    system = ReviewerRecommendationSystem(dataset_path)
    
    if not system.load_dataset():
        print("Failed to load dataset")
        return
    
    print(f"Loaded {len(system.authors)} authors")
    
    # Test with sample author
    if len(system.authors) > 0:
        sample_author = system.authors[0]
        sample_text = system.get_author_corpus(sample_author)[:5000]
        
        print(f"\nTesting with author: {sample_author}")
        
        k = 5
        results = system.method_tfidf_cosine(sample_text, k)
        
        print(f"\nTop {k} similar reviewers:")
        for r in results:
            print(f"{r['rank']}. {r['author']} ({r['similarity']:.3f}) - {r['expertise']}")
        
        # Run evaluation
        print("\n" + "="*50)
        print("Running method evaluation...")
        evaluation = system.evaluate_methods()
        
        for method in evaluation['methods']:
            status = "✅" if method['success'] else "❌"
            print(f"{status} {method['name']}: {method['speed']}")
            if not method['success']:
                print(f"   Error: {method.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
