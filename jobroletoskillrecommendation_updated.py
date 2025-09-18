# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
import re
from sentence_transformers import SentenceTransformer
import pickle
import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional    
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Job Role to Skill Recommendation API")

class JobRoleSkillRecommender:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.data = None
        self.job_roles = None
        self.skills = None
        self.job_role_embeddings = None
        self.skill_embeddings = None
        self.content_similarity_matrix = None
        self.collab_model = None
        self.interaction_matrix = None
        self.role_encoder = LabelEncoder()
        self.skill_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        self.dept_encoder = LabelEncoder()
        self.skill_cat_encoder = LabelEncoder()
        self.skill_type_encoder = LabelEncoder()
        self.proficiency_encoder = LabelEncoder()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.is_trained = False
        
    # Helper methods for feature engineering:

    def _clean_text(self, text):
        """
        Comprehensive text cleaning for embedding preparation
        """
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits (keep letters, spaces, and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _process_field(self, field_data):
        """
        Handle fields that might be stored as lists, arrays, or comma-separated values
        Returns a space-separated string of unique values
        """
        if pd.isna(field_data):
            return ""
        
        # Convert to string first
        field_str = str(field_data)
        
        # Check if it looks like a list/array (common in JSON/DB exports)
        if (field_str.startswith('[') and field_str.endswith(']')) or \
        (field_str.startswith('(') and field_str.endswith(')')) or \
        (',' in field_str and any(char.isalpha() for char in field_str)):
            
            # Clean and split by common delimiters
            cleaned = re.sub(r'[\[\]\(\)\"\']', '', field_str)
            items = [item.strip() for item in cleaned.split(',') if item.strip()]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_items = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            
            return ' '.join(unique_items)
        
        return field_str

    def _add_tfidf_weighting(self):
        """
        Enhance text fields with TF-IDF weighted terms to emphasize important terms
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # For job roles - get important terms
        job_corpus = self.data['Job_Role_Text'].tolist()
        job_tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        job_tfidf.fit(job_corpus)
        
        # For skills - get important terms  
        skill_corpus = self.data['Skill_Text'].tolist()
        skill_tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        skill_tfidf.fit(skill_corpus)
        
        # Store for potential use in analysis or debugging
        self.job_important_terms = job_tfidf.get_feature_names_out()
        self.skill_important_terms = skill_tfidf.get_feature_names_out()
        
        # Optional: You could use these to enhance the text, but for now just store for analysis
        print(f"Top job role terms: {list(self.job_important_terms[:10])}")
        print(f"Top skill terms: {list(self.skill_important_terms[:10])}")
    
    def ingest_data(self, file_path: str):
        """Ingest data from Excel file"""
        try:
            self.data = pd.read_excel(file_path)
            print(f"Data ingested successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error ingesting data: {e}")
            return False
    
    def preprocess_data(self):
        """Clean, deduplicate, and handle missing values"""
        if self.data is None:
            raise ValueError("No data to preprocess. Please ingest data first.")
        
        # Remove duplicates
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_count - len(self.data)} duplicate rows.")
        
        # Handle missing values
        missing_counts = self.data.isnull().sum()
        if missing_counts.any():
            print("Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  - {col}: {count} missing values")
            
            # Fill missing text fields with empty strings
            text_cols = self.data.select_dtypes(include=['object']).columns
            self.data[text_cols] = self.data[text_cols].fillna('')
            
            # For numeric columns, fill with median
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        
        # Normalize text fields
        text_cols = self.data.select_dtypes(include=['object']).columns
        for col in text_cols:
            self.data[col] = self.data[col].astype(str).str.lower().str.strip()
            # Remove extra spaces
            self.data[col] = self.data[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        print("Data preprocessing completed.")
        return True
    
    def feature_engineering(self):
        """Generate entity lists and encode categorical features with enhanced text processing"""
        if self.data is None:
            raise ValueError("No data for feature engineering. Please preprocess data first.")
        
        # Generate unique entities
        self.job_roles = self.data['Job Role'].unique().tolist()
        self.skills = self.data['Skill Title'].unique().tolist()
        
        print(f"Found {len(self.job_roles)} unique job roles and {len(self.skills)} unique skills.")
        
        # Encode categorical features
        categorical_cols = ['Industry', 'Department', 'Skill Category', 'Skill Type', 'Proficiency Level']
        
        # Check which columns exist in the data
        available_cols = [col for col in categorical_cols if col in self.data.columns]
        
        for col in available_cols:
            encoder_name = f"{col.lower().replace(' ', '_')}_encoder"
            if hasattr(self, encoder_name):
                encoder = getattr(self, encoder_name)
                self.data[f"{col}_encoded"] = encoder.fit_transform(self.data[col])
        
        # Create concatenated text fields for embeddings with comprehensive processing
        def create_job_role_text(row):
            parts = []
            
            # Process Job Role (handle potential list data and clean)
            job_role = self._process_field(row['Job Role'])
            job_role_clean = self._clean_text(job_role)
            if job_role_clean:
                parts.append(job_role_clean)
            
            # Add contextual information in a natural language format
            if 'Industry' in row and pd.notna(row['Industry']) and str(row['Industry']).strip():
                industry_clean = self._clean_text(str(row['Industry']))
                if industry_clean:
                    parts.append(f"in {industry_clean} industry")
            
            if 'Department' in row and pd.notna(row['Department']) and str(row['Department']).strip():
                dept_clean = self._clean_text(str(row['Department']))
                if dept_clean:
                    parts.append(f"{dept_clean} department")
            
            # Create natural sounding text
            if len(parts) > 1:
                # If we have context, format it naturally
                return parts[0] + " " + " ".join(parts[1:])
            elif parts:
                return parts[0]
            else:
                return ""
        
        def create_skill_text(row):
            parts = []
            
            # Process Skill Title (handle potential list data and clean)
            skill_title = self._process_field(row['Skill Title'])
            skill_clean = self._clean_text(skill_title)
            if skill_clean:
                parts.append(skill_clean)
            
            # Add contextual information in a meaningful way
            context_fields = [
                ('Skill Category', 'category'),
                ('Skill Type', 'type'), 
                ('Proficiency Level', 'proficiency level')
            ]
            
            for field, description in context_fields:
                if field in row and pd.notna(row[field]) and str(row[field]).strip():
                    field_text = self._clean_text(str(row[field]))
                    if field_text:
                        parts.append(f"{description}: {field_text}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_parts = []
            for part in parts:
                if part not in seen:
                    seen.add(part)
                    unique_parts.append(part)
            
            return ". ".join(unique_parts)
        
        # Apply the enhanced text creation
        self.data['Job_Role_Text'] = self.data.apply(create_job_role_text, axis=1)
        self.data['Skill_Text'] = self.data.apply(create_skill_text, axis=1)
        
        # Final cleaning pass to ensure quality
        self.data['Job_Role_Text'] = self.data['Job_Role_Text'].apply(
            lambda x: self._clean_text(x) if pd.notna(x) else x
        )
        self.data['Skill_Text'] = self.data['Skill_Text'].apply(
            lambda x: self._clean_text(x) if pd.notna(x) else x
        )
        
        # Handle empty results with fallbacks
        job_role_empty_mask = (self.data['Job_Role_Text'].isna()) | (self.data['Job_Role_Text'].str.strip() == '')
        skill_empty_mask = (self.data['Skill_Text'].isna()) | (self.data['Skill_Text'].str.strip() == '')
        
        if job_role_empty_mask.any():
            print(f"Warning: {job_role_empty_mask.sum()} job role text fields are empty, applying fallbacks")
            self.data.loc[job_role_empty_mask, 'Job_Role_Text'] = self.data.loc[job_role_empty_mask, 'Job Role'].apply(
                lambda x: self._clean_text(str(x)) if pd.notna(x) else "unknown job role"
            )
        
        if skill_empty_mask.any():
            print(f"Warning: {skill_empty_mask.sum()} skill text fields are empty, applying fallbacks")
            self.data.loc[skill_empty_mask, 'Skill_Text'] = self.data.loc[skill_empty_mask, 'Skill Title'].apply(
                lambda x: self._clean_text(str(x)) if pd.notna(x) else "unknown skill"
            )
        
        # Add TF-IDF informed weighting (optional enhancement)
        try:
            self._add_tfidf_weighting()
            print("TF-IDF weighting applied to enhance text features")
        except Exception as e:
            print(f"TF-IDF weighting skipped due to error: {e}")
        
        print("Enhanced feature engineering completed.")
        print(f"Sample Job Role Text: {self.data['Job_Role_Text'].iloc[0][:100]}...")
        print(f"Sample Skill Text: {self.data['Skill_Text'].iloc[0][:100]}...")
        
        return True
    
    def create_embeddings(self):
        """Create embeddings for job roles and skills with enhanced processing"""
        if self.data is None:
            raise ValueError("No data for embedding creation. Please run feature engineering first.")
        
        # Get unique job roles and their text representations
        job_role_data = self.data[['Job Role', 'Job_Role_Text']].drop_duplicates()
        skill_data = self.data[['Skill Title', 'Skill_Text']].drop_duplicates()
        
        # Validate text quality before embedding and track valid entries
        valid_job_roles = []
        job_role_texts = []
        for _, row in job_role_data.iterrows():
            text = row['Job_Role_Text']
            if pd.notna(text) and text.strip() and len(text.split()) >= 1:  # At least one word
                valid_job_roles.append(row['Job Role'])
                job_role_texts.append(text)
            else:
                print(f"Warning: Skipping invalid job role text: '{text}'")
        
        valid_skills = []
        skill_texts = []
        for _, row in skill_data.iterrows():
            text = row['Skill_Text']
            if pd.notna(text) and text.strip() and len(text.split()) >= 1:  # At least one word
                valid_skills.append(row['Skill Title'])
                skill_texts.append(text)
            else:
                print(f"Warning: Skipping invalid skill text: '{text}'")
        
        # Create embeddings with better parameters
        self.job_role_embeddings = self.embedding_model.encode(
            job_role_texts, 
            show_progress_bar=True,
            batch_size=32,  # Optimal batch size for better performance
            convert_to_tensor=True  # Better for similarity calculations
        )
        
        self.skill_embeddings = self.embedding_model.encode(
            skill_texts, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=True
        )
        
        # Create mapping dictionaries using only valid entries that have embeddings
        self.job_role_to_idx = {role: idx for idx, role in enumerate(valid_job_roles)}
        self.idx_to_job_role = {idx: role for role, idx in self.job_role_to_idx.items()}
        
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(valid_skills)}
        self.idx_to_skill = {idx: skill for skill, idx in self.skill_to_idx.items()}
        
        # Store the actual text used for embeddings for debugging
        self.job_role_embedding_texts = job_role_texts
        self.skill_embedding_texts = skill_texts
        
        print(f"Embeddings created successfully. Job roles: {len(job_role_texts)}, Skills: {len(skill_texts)}")
        return True
    
    def compute_content_similarity(self):
        """Compute cosine similarity between job role and skill embeddings with enhancements"""
        if self.job_role_embeddings is None or self.skill_embeddings is None:
            raise ValueError("Embeddings not found. Please create embeddings first.")
        
        # Convert tensors to numpy if needed
        if hasattr(self.job_role_embeddings, 'numpy'):
            job_embeddings = self.job_role_embeddings.numpy()
            skill_embeddings = self.skill_embeddings.numpy()
        else:
            job_embeddings = self.job_role_embeddings
            skill_embeddings = self.skill_embeddings
        
        # Normalize embeddings for better cosine similarity
        from sklearn.preprocessing import normalize
        job_embeddings_norm = normalize(job_embeddings)
        skill_embeddings_norm = normalize(skill_embeddings)
        
        self.content_similarity_matrix = cosine_similarity(
            job_embeddings_norm, 
            skill_embeddings_norm
        )
        
        # Add similarity matrix diagnostics
        avg_similarity = np.mean(self.content_similarity_matrix)
        max_similarity = np.max(self.content_similarity_matrix)
        min_similarity = np.min(self.content_similarity_matrix)
        
        print(f"Content similarity matrix computed. Avg: {avg_similarity:.3f}, Range: [{min_similarity:.3f}, {max_similarity:.3f}]")
        
        # Store similarity statistics for recommendation thresholding
        self.similarity_stats = {
            'mean': avg_similarity,
            'std': np.std(self.content_similarity_matrix),
            'max': max_similarity,
            'min': min_similarity
        }
        
        return True
    
    def prepare_collaborative_data(self):
        """Construct JobRole-Skill interaction matrix with strict mapping"""
        if self.data is None:
            raise ValueError("No data for collaborative filtering. Please preprocess data first.")
        
        # Build interaction counts
        interactions = self.data.groupby(['Job Role', 'Skill Title']).size().reset_index(name='count')

        # Map using embedding dictionaries
        interactions['role_idx'] = interactions['Job Role'].map(self.job_role_to_idx)
        interactions['skill_idx'] = interactions['Skill Title'].map(self.skill_to_idx)

        # 🔹 Drop rows where mapping failed
        initial_count = len(interactions)
        interactions = interactions.dropna(subset=['role_idx', 'skill_idx'])
        dropped = initial_count - len(interactions)
        if dropped > 0:
            print(f"Warning: Dropped {dropped} unmapped interactions")

        if len(interactions) == 0:
            print("Error: No valid interactions after filtering. Collaborative filtering cannot proceed.")
            return False

        # 🔹 Ensure safe integer indices
        interactions['role_idx'] = interactions['role_idx'].astype(int)
        interactions['skill_idx'] = interactions['skill_idx'].astype(int)

        # 🔹 Recheck bounds
        n_roles = len(self.job_role_to_idx)
        n_skills = len(self.skill_to_idx)
        if interactions['role_idx'].max() >= n_roles or interactions['skill_idx'].max() >= n_skills:
            print("Error: Found indices out of bounds after filtering. Skipping collaborative step.")
            return False

        # Build sparse matrix
        self.interaction_matrix = csr_matrix(
            (interactions['count'], (interactions['role_idx'], interactions['skill_idx'])),
            shape=(n_roles, n_skills)
        )
        print(f"Collaborative filtering data prepared. Matrix: {n_roles} roles x {n_skills} skills")
        return True

    
    def train_collaborative_model(self, model_type='nmf', n_components=50):
        """Train collaborative filtering model with enhanced validation"""
        if self.interaction_matrix is None:
            print("No interaction matrix available for collaborative filtering")
            return False
        
        # Check if matrix is too sparse for meaningful training
        if self.interaction_matrix.nnz < n_components * 10:
            print("Warning: Interaction matrix may be too sparse for effective collaborative filtering")
            return False
        
        try:
            if model_type == 'nmf':
                self.collab_model = NMF(
                    n_components=n_components, 
                    init='nndsvd',
                    random_state=42,
                    max_iter=500,
                    alpha=0.1
                )
                
                W = self.collab_model.fit_transform(self.interaction_matrix)
                H = self.collab_model.components_
                
                # Reconstruct the matrix for recommendations
                self.collab_similarity_matrix = np.dot(W, H)
                
                # Calculate reconstruction error
                reconstruction_error = np.linalg.norm(self.interaction_matrix - np.dot(W, H))
                print(f"NMF model trained. Reconstruction error: {reconstruction_error:.4f}")
                
                return True
                
        except Exception as e:
            print(f"Collaborative model training failed: {e}")
            return False
    
    
    def generate_content_recommendations(self, job_role, top_n=5, min_similarity=0.1):
         if self.content_similarity_matrix is None:
             raise ValueError("Content similarity matrix not computed. Please compute it first.")

         if job_role not in self.job_role_to_idx:
             raise ValueError(f"Job role '{job_role}' not found in index mapping.")

         role_idx = self.job_role_to_idx[job_role]
         similarities = self.content_similarity_matrix[role_idx]

         # Get top N indices
         top_indices = np.argsort(similarities)[::-1][:top_n * 2]  # take extra in case of invalid indices

         recommendations = []
         for idx in top_indices:
             if idx in self.idx_to_skill:  # ✅ guard clause
                 recommendations.append({
                     'skill': self.idx_to_skill[idx],
                     'score': float(similarities[idx]),
                     'confidence': 'high' if similarities[idx] > self.similarity_stats['mean'] + self.similarity_stats['std'] else 'medium'
                 })
             if len(recommendations) >= top_n:
                 break

         # ✅ fallback if no valid skills found
         if not recommendations:
             recommendations = [{"skill": "No valid skills found", "score": 0.0, "confidence": "low"}]

         return recommendations


    def _find_similar_job_roles(self, query_role, threshold=0.7):
        """Fuzzy match for similar job roles"""
        from fuzzywuzzy import process
        
        try:
            matches = process.extract(query_role, self.job_roles, limit=3)
            return [match[0] for match in matches if match[1] >= threshold * 100]
        except:
            return []

    def _handle_similar_job_roles(self, original_role, similar_roles, top_n):
        """Handle recommendations for similar job roles"""
        all_recommendations = []
        
        for similar_role in similar_roles:
            recs = self.generate_content_recommendations(similar_role, top_n=top_n)
            for rec in recs:
                rec['source_role'] = similar_role
                rec['original_query'] = original_role
            all_recommendations.extend(recs)
        
        # Deduplicate and sort
        seen_skills = set()
        unique_recommendations = []
        
        for rec in sorted(all_recommendations, key=lambda x: x['score'], reverse=True):
            if rec['skill'] not in seen_skills:
                seen_skills.add(rec['skill'])
                unique_recommendations.append(rec)
            if len(unique_recommendations) >= top_n:
                break
        
        return unique_recommendations
    
    def generate_collaborative_recommendations(self, job_role, top_n=5):
        """Generate collaborative filtering recommendations for a job role"""
        if self.collab_similarity_matrix is None:
            raise ValueError("Collaborative similarity matrix not found. Please train the model first.")
        
        if job_role not in self.job_role_to_idx:
            raise ValueError(f"Job role '{job_role}' not found in the data.")
        
        role_idx = self.job_role_to_idx[job_role]
        similarities = self.collab_similarity_matrix[role_idx]
        
        # Get top N skills
        top_skill_indices = np.argsort(similarities)[::-1][:top_n]
        recommendations = [
            {
                'skill': self.idx_to_skill[idx],
                'score': float(similarities[idx])
            }
            for idx in top_skill_indices
        ]
        
        return recommendations
    
    def generate_hybrid_recommendations(self, job_role, top_n=5, content_weight=0.5, collab_weight=0.5):
        """Generate hybrid recommendations using weighted combination"""
        if self.content_similarity_matrix is None or self.collab_similarity_matrix is None:
            raise ValueError("Both content and collaborative matrices are required for hybrid recommendations.")
        
        if job_role not in self.job_role_to_idx:
            raise ValueError(f"Job role '{job_role}' not found in the data.")
        
        role_idx = self.job_role_to_idx[job_role]
        
        # Get scores from both approaches
        content_scores = self.content_similarity_matrix[role_idx]
        collab_scores = self.collab_similarity_matrix[role_idx]
        
        # Normalize scores
        content_scores_norm = (content_scores - np.min(content_scores)) / (np.max(content_scores) - np.min(content_scores))
        collab_scores_norm = (collab_scores - np.min(collab_scores)) / (np.max(collab_scores) - np.min(collab_scores))
        
        # Combine scores
        hybrid_scores = (content_weight * content_scores_norm) + (collab_weight * collab_scores_norm)
        
        # Get top N skills
        top_skill_indices = np.argsort(hybrid_scores)[::-1][:top_n]
        recommendations = [
            {
                'skill': self.idx_to_skill[idx],
                'score': float(hybrid_scores[idx]),
                'content_score': float(content_scores_norm[idx]),
                'collab_score': float(collab_scores_norm[idx])
            }
            for idx in top_skill_indices
        ]
        
        return recommendations
    
    def train_full_pipeline(self, file_path):
        """Run the complete training pipeline with validation and monitoring"""
        print("Starting enhanced training pipeline...")
        
        try:
            # Step 1: Ingest data
            print("Step 1: Ingesting data...")
            if not self.ingest_data(file_path):
                raise Exception("Data ingestion failed")
            
            # Step 2: Preprocess data
            print("Step 2: Preprocessing data...")
            if not self.preprocess_data():
                raise Exception("Data preprocessing failed")
            
            # Step 3: Feature engineering
            print("Step 3: Feature engineering...")
            if not self.feature_engineering():
                raise Exception("Feature engineering failed")
            
            # Step 4: Create embeddings
            print("Step 4: Creating embeddings...")
            if not self.create_embeddings():
                raise Exception("Embedding creation failed")
            
            # Step 5: Compute content similarity
            print("Step 5: Computing content similarity...")
            if not self.compute_content_similarity():
                raise Exception("Similarity computation failed")
            
            # Step 6: Prepare collaborative data
            print("Step 6: Preparing collaborative data...")
            collaborative_success = self.prepare_collaborative_data()
            
            # Step 7: Train collaborative model (only if collaborative data was prepared successfully)
            if collaborative_success:
                print("Step 7: Training collaborative model...")
                collaborative_training_success = self.train_collaborative_model()
                
                if not collaborative_training_success:
                    print("Collaborative model training failed, using content-based only")
                    self.collab_similarity_matrix = np.zeros_like(self.content_similarity_matrix)
            else:
                print("Collaborative data preparation failed, using content-based only")
                self.collab_similarity_matrix = np.zeros_like(self.content_similarity_matrix)
            
            self.is_trained = True
            print("Enhanced training pipeline completed successfully.")
            
            # Generate sample recommendations for validation
            if self.job_roles:
                sample_role = self.job_roles[0]
                sample_recs = self.generate_content_recommendations(sample_role, top_n=3)
                print(f"Sample recommendations for '{sample_role}': {[r['skill'] for r in sample_recs]}")
            
            return True
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
            return False
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

# Initialize the recommender system
recommender = JobRoleSkillRecommender()

# FastAPI endpoints
@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...)):
    """Endpoint to upload new data"""
    try:
        # Save the uploaded file
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Train the model with the new data
        success = recommender.train_full_pipeline(file_path)
        
        if success:
            return JSONResponse(
                content={"message": "Data uploaded and model trained successfully"},
                status_code=200
            )
        else:
            return JSONResponse(
                content={"message": "Error processing data"},
                status_code=500
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/{job_role}")
async def get_recommendations(
    job_role: str, 
    top_n: int = 5, 
    method: str = "hybrid",
    content_weight: float = 0.5,
    collab_weight: float = 0.5
):
    """Endpoint to get recommendations for a job role"""
    try:
        if not recommender.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained. Please upload data first.")
        
        if method == "content":
            recommendations = recommender.generate_content_recommendations(job_role, top_n)
        elif method == "collaborative":
            recommendations = recommender.generate_collaborative_recommendations(job_role, top_n)
        elif method == "hybrid":
            recommendations = recommender.generate_hybrid_recommendations(
                job_role, top_n, content_weight, collab_weight
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'content', 'collaborative', or 'hybrid'.")
        
        return {
            "job_role": job_role,
            "recommendations": recommendations,
            "method": method
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")
async def retrain_model():
    """Endpoint to trigger model retraining"""
    try:
        if recommender.data is None:
            raise HTTPException(status_code=400, detail="No data available for retraining.")
        
        # Retrain the model with current data
        success = recommender.train_full_pipeline("current_data.xlsx")
        
        if success:
            return {"message": "Model retrained successfully"}
        else:
            return {"message": "Error retraining model"}, 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job_roles/")
async def get_job_roles():
    """Endpoint to get all available job roles"""
    try:
        if recommender.job_roles is None:
            raise HTTPException(status_code=400, detail="No job roles available. Please upload data first.")
        
        return {"job_roles": recommender.job_roles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_trained": recommender.is_trained}

if __name__ == "__main__":
    # For demonstration purposes, we'll create a sample data file if it doesn't exist
    # In a real scenario, you would use your actual JobRoleToSkillRecommendation.xlsx file
    import os
    if not os.path.exists("JobRoleToSkillRecommendation.xlsx"):
        print("Creating sample data file for demonstration...")
        sample_data = {
            'Job Role': ['Data Scientist', 'Data Scientist', 'ML Engineer', 'ML Engineer', 'Data Analyst', 'Data Analyst'],
            'Industry': ['Tech', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance'],
            'Department': ['Engineering', 'Engineering', 'Engineering', 'Engineering', 'Analytics', 'Analytics'],
            'Skill': ['Python', 'Machine Learning', 'Python', 'Deep Learning', 'SQL', 'Data Visualization'],
            'Skill Category': ['Programming', 'ML', 'Programming', 'ML', 'Database', 'Visualization'],
            'Skill Type': ['Technical', 'Technical', 'Technical', 'Technical', 'Technical', 'Technical'],
            'Proficiency Level': ['Advanced', 'Intermediate', 'Advanced', 'Intermediate', 'Intermediate', 'Intermediate']
        }
        df = pd.DataFrame(sample_data)
        df.to_excel("JobRoleToSkillRecommendation.xlsx", index=False)
        print("Sample data file created.")
    
    # Train the model with the sample data
    print("Training model with sample data...")
    success = recommender.train_full_pipeline("JobRoleToSkillRecommendation.xlsx")
    
    if success:
        print("Model training successful. Starting API server...")
        uvicorn.run(app, host="0.0.0.0", port=5000)
    else:
        print("Model training failed. Please check your data file.")