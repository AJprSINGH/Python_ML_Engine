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

    # ---------------------------
    # Helper text functions
    # ---------------------------
    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _process_field(self, field_data):
        if pd.isna(field_data):
            return ""
        field_str = str(field_data)
        if (field_str.startswith('[') and field_str.endswith(']')) or \
           (field_str.startswith('(') and field_str.endswith(')')) or \
           (',' in field_str and any(char.isalpha() for char in field_str)):
            cleaned = re.sub(r'[\[\]\(\)\"\']', '', field_str)
            items = [item.strip() for item in cleaned.split(',') if item.strip()]
            seen = set()
            unique_items = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            return ' '.join(unique_items)
        return field_str

    def _add_tfidf_weighting(self):
        job_corpus = self.data['Job_Role_Text'].tolist()
        job_tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        job_tfidf.fit(job_corpus)
        skill_corpus = self.data['Skill_Text'].tolist()
        skill_tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        skill_tfidf.fit(skill_corpus)
        self.job_important_terms = job_tfidf.get_feature_names_out()
        self.skill_important_terms = skill_tfidf.get_feature_names_out()
        print(f"Top job role terms: {list(self.job_important_terms[:10])}")
        print(f"Top skill terms: {list(self.skill_important_terms[:10])}")

    # ---------------------------
    # Data ingestion & preprocess
    # ---------------------------
    def ingest_data(self, file_path: str):
        try:
            self.data = pd.read_excel(file_path)
            print(f"Data ingested successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error ingesting data: {e}")
            return False

    def preprocess_data(self):
        if self.data is None:
            raise ValueError("No data to preprocess. Please ingest data first.")
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_count - len(self.data)} duplicate rows.")
        missing_counts = self.data.isnull().sum()
        if missing_counts.any():
            print("Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  - {col}: {count} missing values")
            text_cols = self.data.select_dtypes(include=['object']).columns
            self.data[text_cols] = self.data[text_cols].fillna('')
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        text_cols = self.data.select_dtypes(include=['object']).columns
        for col in text_cols:
            self.data[col] = self.data[col].astype(str).str.lower().str.strip()
            self.data[col] = self.data[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        print("Data preprocessing completed.")
        return True

    # ---------------------------
    # Feature engineering
    # ---------------------------
    def feature_engineering(self):
        if self.data is None:
            raise ValueError("No data for feature engineering. Please preprocess data first.")

        # Normalize fields to avoid string mismatches
        if 'Job Role' not in self.data.columns or 'Skill Title' not in self.data.columns:
            raise ValueError("Expected columns 'Job Role' and 'Skill Title' in input data.")

        self.data['Job Role'] = self.data['Job Role'].astype(str).str.lower().str.strip()
        self.data['Skill Title'] = self.data['Skill Title'].astype(str).str.lower().str.strip()

        self.job_roles = self.data['Job Role'].unique().tolist()
        self.skills = self.data['Skill Title'].unique().tolist()
        print(f"Found {len(self.job_roles)} unique job roles and {len(self.skills)} unique skills.")

        # Encode categorical columns if present
        categorical_cols = ['Industry', 'Department', 'Skill Category', 'Skill Type', 'Proficiency Level']
        available_cols = [col for col in categorical_cols if col in self.data.columns]
        for col in available_cols:
            encoder_name = f"{col.lower().replace(' ', '_')}_encoder"
            if hasattr(self, encoder_name):
                encoder = getattr(self, encoder_name)
                self.data[f"{col}_encoded"] = encoder.fit_transform(self.data[col].astype(str))

        # Create text for embeddings
        def create_job_role_text(row):
            parts = []
            job_role = self._process_field(row['Job Role'])
            job_role_clean = self._clean_text(job_role)
            if job_role_clean:
                parts.append(job_role_clean)
            if 'Industry' in row and pd.notna(row['Industry']) and str(row['Industry']).strip():
                industry_clean = self._clean_text(str(row['Industry']))
                if industry_clean:
                    parts.append(f"in {industry_clean} industry")
            if 'Department' in row and pd.notna(row['Department']) and str(row['Department']).strip():
                dept_clean = self._clean_text(str(row['Department']))
                if dept_clean:
                    parts.append(f"{dept_clean} department")
            if len(parts) > 1:
                return parts[0] + " " + " ".join(parts[1:])
            elif parts:
                return parts[0]
            else:
                return ""

        def create_skill_text(row):
            parts = []
            skill_title = self._process_field(row['Skill Title'])
            skill_clean = self._clean_text(skill_title)
            if skill_clean:
                parts.append(skill_clean)
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
            seen = set()
            unique_parts = []
            for part in parts:
                if part not in seen:
                    seen.add(part)
                    unique_parts.append(part)
            return ". ".join(unique_parts)

        self.data['Job_Role_Text'] = self.data.apply(create_job_role_text, axis=1)
        self.data['Skill_Text'] = self.data.apply(create_skill_text, axis=1)

        self.data['Job_Role_Text'] = self.data['Job_Role_Text'].apply(lambda x: self._clean_text(x) if pd.notna(x) else "")
        self.data['Skill_Text'] = self.data['Skill_Text'].apply(lambda x: self._clean_text(x) if pd.notna(x) else "")

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

        try:
            self._add_tfidf_weighting()
            print("TF-IDF weighting applied to enhance text features")
        except Exception as e:
            print(f"TF-IDF weighting skipped due to error: {e}")

        print("Enhanced feature engineering completed.")
        return True

    # ---------------------------
    # Embeddings
    # ---------------------------
    def create_embeddings(self):
        if self.data is None:
            raise ValueError("No data for embedding creation. Please run feature engineering first.")
        job_role_data = self.data[['Job Role', 'Job_Role_Text']].drop_duplicates()
        skill_data = self.data[['Skill Title', 'Skill_Text']].drop_duplicates()

        valid_job_roles = []
        job_role_texts = []
        for _, row in job_role_data.iterrows():
            text = row['Job_Role_Text']
            if pd.notna(text) and text.strip():
                valid_job_roles.append(row['Job Role'])
                job_role_texts.append(text)
            else:
                print(f"Warning: Skipping invalid job role text: '{text}'")

        valid_skills = []
        skill_texts = []
        for _, row in skill_data.iterrows():
            text = row['Skill_Text']
            if pd.notna(text) and text.strip():
                valid_skills.append(row['Skill Title'])
                skill_texts.append(text)
            else:
                print(f"Warning: Skipping invalid skill text: '{text}'")

        # create embeddings
        self.job_role_embeddings = self.embedding_model.encode(
            job_role_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=True
        )
        self.skill_embeddings = self.embedding_model.encode(
            skill_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=True
        )

        # mappings reflect only embedded entries
        self.job_role_to_idx = {role: idx for idx, role in enumerate(valid_job_roles)}
        self.idx_to_job_role = {idx: role for role, idx in self.job_role_to_idx.items()}
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(valid_skills)}
        self.idx_to_skill = {idx: skill for skill, idx in self.skill_to_idx.items()}

        # store raw texts
        self.job_role_embedding_texts = job_role_texts
        self.skill_embedding_texts = skill_texts

        print(f"Embeddings created successfully. Job roles: {len(job_role_texts)}, Skills: {len(skill_texts)}")
        return True

    # ---------------------------
    # Content similarity
    # ---------------------------
    def compute_content_similarity(self):
        if self.job_role_embeddings is None or self.skill_embeddings is None:
            raise ValueError("Embeddings not found. Please create embeddings first.")
        if hasattr(self.job_role_embeddings, 'numpy'):
            job_embeddings = self.job_role_embeddings.numpy()
            skill_embeddings = self.skill_embeddings.numpy()
        else:
            job_embeddings = self.job_role_embeddings
            skill_embeddings = self.skill_embeddings
        from sklearn.preprocessing import normalize
        job_embeddings_norm = normalize(job_embeddings)
        skill_embeddings_norm = normalize(skill_embeddings)
        self.content_similarity_matrix = cosine_similarity(job_embeddings_norm, skill_embeddings_norm)
        avg_similarity = np.mean(self.content_similarity_matrix)
        max_similarity = np.max(self.content_similarity_matrix)
        min_similarity = np.min(self.content_similarity_matrix)
        print(f"Content similarity matrix computed. Avg: {avg_similarity:.3f}, Range: [{min_similarity:.3f}, {max_similarity:.3f}]")
        self.similarity_stats = {
            'mean': avg_similarity,
            'std': np.std(self.content_similarity_matrix),
            'max': max_similarity,
            'min': min_similarity
        }
        return True

    # ---------------------------
    # Collaborative data
    # ---------------------------
    def prepare_collaborative_data(self):
        """Construct JobRole-Skill interaction matrix with strict mapping"""
        if self.data is None:
            raise ValueError("No data for collaborative filtering. Please preprocess data first.")

        interactions = self.data.groupby(['Job Role', 'Skill Title']).size().reset_index(name='count')

        # If proficiency exists, add weight column (optional)
        if 'Proficiency Level' in self.data.columns:
            proficiency_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
            prof = self.data.groupby(['Job Role', 'Skill Title'])['Proficiency Level'].first().reset_index()
            interactions = interactions.merge(prof, on=['Job Role', 'Skill Title'], how='left')
            interactions['weight'] = interactions['Proficiency Level'].map(
                lambda x: proficiency_map.get(str(x).lower(), 1) if pd.notna(x) else 1
            )
            interactions['weighted_count'] = interactions['count'] * interactions['weight']
            counts_col = 'weighted_count'
        else:
            interactions['weighted_count'] = interactions['count']
            counts_col = 'weighted_count'

        # Map to embedding indices (only embedded entries will map)
        interactions['role_idx'] = interactions['Job Role'].map(self.job_role_to_idx)
        interactions['skill_idx'] = interactions['Skill Title'].map(self.skill_to_idx)

        # Drop rows with missing mappings
        initial_count = len(interactions)
        interactions = interactions.dropna(subset=['role_idx', 'skill_idx'])
        dropped = initial_count - len(interactions)
        if dropped > 0:
            print(f"Warning: Dropped {dropped} unmapped interactions")

        if len(interactions) == 0:
            print("Error: No valid interactions after filtering. Collaborative filtering cannot proceed.")
            return False

        # Ensure indices are ints
        interactions['role_idx'] = interactions['role_idx'].astype(int)
        interactions['skill_idx'] = interactions['skill_idx'].astype(int)

        n_roles = len(self.job_role_to_idx)
        n_skills = len(self.skill_to_idx)

        # Re-check bounds after filtering
        if interactions['role_idx'].max() >= n_roles or interactions['skill_idx'].max() >= n_skills:
            print("Error: Found indices out of bounds after filtering. Skipping collaborative step.")
            # Optional: print some offending rows for debugging
            oob_roles = interactions[interactions['role_idx'] >= n_roles]
            oob_skills = interactions[interactions['skill_idx'] >= n_skills]
            if len(oob_roles) > 0:
                print("Sample out-of-bounds role rows (first 5):")
                print(oob_roles.head(5))
            if len(oob_skills) > 0:
                print("Sample out-of-bounds skill rows (first 5):")
                print(oob_skills.head(5))
            return False

        # Build sparse interaction matrix
        self.interaction_matrix = csr_matrix(
            (interactions[counts_col], (interactions['role_idx'], interactions['skill_idx'])),
            shape=(n_roles, n_skills)
        )

        matrix_density = (self.interaction_matrix.nnz / (n_roles * n_skills)) * 100
        print(f"Collaborative filtering data prepared. Density: {matrix_density:.4f}%")
        return True

    # ---------------------------
    # Collaborative model training
    # ---------------------------
    def train_collaborative_model(self, model_type='nmf', n_components=50):
        if self.interaction_matrix is None:
            print("No interaction matrix available for collaborative filtering")
            return False
        if self.interaction_matrix.nnz < n_components * 10:
            print("Warning: Interaction matrix may be too sparse for effective collaborative filtering")
            return False
        try:
            if model_type == 'nmf':
                self.collab_model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500, alpha=0.1)
                W = self.collab_model.fit_transform(self.interaction_matrix)
                H = self.collab_model.components_
                self.collab_similarity_matrix = np.dot(W, H)
                reconstruction_error = np.linalg.norm(self.interaction_matrix - np.dot(W, H))
                print(f"NMF model trained. Reconstruction error: {reconstruction_error:.4f}")
                return True
        except Exception as e:
            print(f"Collaborative model training failed: {e}")
            return False

    # ---------------------------
    # Recommendation generators
    # ---------------------------
    def generate_content_recommendations(self, job_role, top_n=5, min_similarity=0.1):
        if self.content_similarity_matrix is None:
            raise ValueError("Content similarity matrix not found. Please compute similarities first.")

        if job_role not in self.job_role_to_idx:
            # try fuzzy match fallback
            similar_roles = self._find_similar_job_roles(job_role)
            if similar_roles:
                return self._handle_similar_job_roles(job_role, similar_roles, top_n)
            raise ValueError(f"Job role '{job_role}' not found in the data.")

        role_idx = self.job_role_to_idx[job_role]
        similarities = self.content_similarity_matrix[role_idx]

        # pick top candidates (take extra to allow skipping)
        candidate_count = max(top_n * 2, top_n + 10)
        top_indices = np.argsort(similarities)[::-1][:candidate_count]

        recommendations = []
        for idx in top_indices:
            if int(idx) in self.idx_to_skill:
                score = float(similarities[idx])
                # apply min_similarity threshold
                if score < min_similarity:
                    continue
                recommendations.append({
                    'skill': self.idx_to_skill[int(idx)],
                    'score': score,
                    'confidence': 'high' if score > self.similarity_stats['mean'] + self.similarity_stats['std'] else 'medium'
                })
            if len(recommendations) >= top_n:
                break

        if not recommendations:
            return [{"skill": "No sufficiently similar skills found", "score": 0.0, "confidence": "low"}]

        return recommendations

    def _find_similar_job_roles(self, query_role, threshold=0.7):
        try:
            from fuzzywuzzy import process
            matches = process.extract(query_role, self.job_roles, limit=5)
            return [match[0] for match in matches if match[1] >= threshold * 100]
        except Exception:
            return []

    def _handle_similar_job_roles(self, original_role, similar_roles, top_n):
        all_recommendations = []
        for similar_role in similar_roles:
            recs = self.generate_content_recommendations(similar_role, top_n=top_n)
            for rec in recs:
                rec['source_role'] = similar_role
                rec['original_query'] = original_role
            all_recommendations.extend(recs)
        seen = set()
        unique_recs = []
        for rec in sorted(all_recommendations, key=lambda x: x['score'], reverse=True):
            if rec['skill'] not in seen:
                seen.add(rec['skill'])
                unique_recs.append(rec)
            if len(unique_recs) >= top_n:
                break
        return unique_recs

    def generate_collaborative_recommendations(self, job_role, top_n=5):
        if self.collab_similarity_matrix is None:
            raise ValueError("Collaborative similarity matrix not found. Please train the model first.")
        if job_role not in self.job_role_to_idx:
            raise ValueError(f"Job role '{job_role}' not found in the data.")
        role_idx = self.job_role_to_idx[job_role]
        similarities = self.collab_similarity_matrix[role_idx]
        top_skill_indices = np.argsort(similarities)[::-1][:top_n]
        recs = []
        for idx in top_skill_indices:
            if int(idx) in self.idx_to_skill:
                recs.append({'skill': self.idx_to_skill[int(idx)], 'score': float(similarities[idx])})
        return recs

    def generate_hybrid_recommendations(self, job_role, top_n=5, content_weight=0.5, collab_weight=0.5):
        if self.content_similarity_matrix is None:
            raise ValueError("Content similarity matrix not computed.")
        if self.collab_similarity_matrix is None:
            raise ValueError("Collaborative similarity matrix not computed.")
        if job_role not in self.job_role_to_idx:
            raise ValueError(f"Job role '{job_role}' not found in the data.")
        role_idx = self.job_role_to_idx[job_role]
        content_scores = self.content_similarity_matrix[role_idx]
        collab_scores = self.collab_similarity_matrix[role_idx]
        # normalize safely
        def safe_norm(x):
            denom = (np.max(x) - np.min(x))
            return (x - np.min(x)) / denom if denom > 0 else np.zeros_like(x)
        content_scores_norm = safe_norm(content_scores)
        collab_scores_norm = safe_norm(collab_scores)
        hybrid_scores = (content_weight * content_scores_norm) + (collab_weight * collab_scores_norm)
        top_skill_indices = np.argsort(hybrid_scores)[::-1][:top_n]
        recs = []
        for idx in top_skill_indices:
            if int(idx) in self.idx_to_skill:
                recs.append({
                    'skill': self.idx_to_skill[int(idx)],
                    'score': float(hybrid_scores[idx]),
                    'content_score': float(content_scores_norm[idx]),
                    'collab_score': float(collab_scores_norm[idx])
                })
        return recs

    # ---------------------------
    # Training pipeline & persistence
    # ---------------------------
    def train_full_pipeline(self, file_path):
        print("Starting enhanced training pipeline...")
        try:
            print("Step 1: Ingesting data...")
            if not self.ingest_data(file_path):
                raise Exception("Data ingestion failed")
            print("Step 2: Preprocessing data...")
            if not self.preprocess_data():
                raise Exception("Data preprocessing failed")
            print("Step 3: Feature engineering...")
            if not self.feature_engineering():
                raise Exception("Feature engineering failed")
            print("Step 4: Creating embeddings...")
            if not self.create_embeddings():
                raise Exception("Embedding creation failed")
            print("Step 5: Computing content similarity...")
            if not self.compute_content_similarity():
                raise Exception("Similarity computation failed")
            print("Step 6: Preparing collaborative data...")
            collaborative_success = self.prepare_collaborative_data()
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

            # sample validation
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
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


# Initialize recommender
recommender = JobRoleSkillRecommender()

# ---------------------------
# FastAPI endpoints
# ---------------------------
@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...)):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        success = recommender.train_full_pipeline(file_path)
        if success:
            return JSONResponse(content={"message": "Data uploaded and model trained successfully"}, status_code=200)
        else:
            return JSONResponse(content={"message": "Error processing data"}, status_code=500)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations")
async def get_recommendations(job_role: str, top_n: int = 5, method: str = "hybrid", content_weight: float = 0.5, collab_weight: float = 0.5):
    try:
        if not recommender.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained. Please upload data first.")
        if method == "content":
            recommendations = recommender.generate_content_recommendations(job_role, top_n)
        elif method == "collaborative":
            recommendations = recommender.generate_collaborative_recommendations(job_role, top_n)
        elif method == "hybrid":
            recommendations = recommender.generate_hybrid_recommendations(job_role, top_n, content_weight, collab_weight)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'content', 'collaborative', or 'hybrid'.")
        return {"job_role": job_role, "recommendations": recommendations, "method": method}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain/")
async def retrain_model():
    try:
        if recommender.data is None:
            raise HTTPException(status_code=400, detail="No data available for retraining.")
        success = recommender.train_full_pipeline("current_data.xlsx")
        if success:
            return {"message": "Model retrained successfully"}
        else:
            return {"message": "Error retraining model"}, 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job_roles/")
async def get_job_roles():
    try:
        if recommender.job_roles is None:
            raise HTTPException(status_code=400, detail="No job roles available. Please upload data first.")
        return {"job_roles": recommender.job_roles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/")
async def health_check():
    return {"status": "healthy", "model_trained": recommender.is_trained}


from pydantic import BaseModel
from typing import List

class SkillGapRequest(BaseModel):
    target_role: str
    current_skills: List[str]
    top_n: int = 10
    method: str = "hybrid"

@app.post("/skill_gap")
async def skill_gap_analysis(request: SkillGapRequest):
    try:
        if not recommender.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained. Please upload data first.")

        # Step 1: Get target role recommendations
        if request.method == "content":
            recommendations = recommender.generate_content_recommendations(request.target_role, request.top_n)
        elif request.method == "collaborative":
            recommendations = recommender.generate_collaborative_recommendations(request.target_role, request.top_n)
        elif request.method == "hybrid":
            recommendations = recommender.generate_hybrid_recommendations(request.target_role, request.top_n)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'content', 'collaborative', or 'hybrid'.")

        # Step 2: Normalize current skills
        current_set = {s.lower().strip() for s in request.current_skills}

        # Step 3: Compare with recommendations
        required_skills = {rec['skill'].lower().strip(): rec for rec in recommendations}
        missing_skills = [rec for skill, rec in required_skills.items() if skill not in current_set]
        matched_skills = [rec for skill, rec in required_skills.items() if skill in current_set]

        return {
            "target_role": request.target_role,
            "current_skills": list(current_set),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "method": request.method
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RoleGapRequest(BaseModel):
    from_role: str
    to_role: str
    top_n: int = 10
    method: str = "hybrid"

@app.post("/role_gap")
async def role_gap_analysis(request: RoleGapRequest):
    try:
        if not recommender.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained. Please upload data first.")

        # Get source role and target role recommendations
        if request.method == "content":
            from_recs = recommender.generate_content_recommendations(request.from_role, request.top_n)
            to_recs = recommender.generate_content_recommendations(request.to_role, request.top_n)
        elif request.method == "collaborative":
            from_recs = recommender.generate_collaborative_recommendations(request.from_role, request.top_n)
            to_recs = recommender.generate_collaborative_recommendations(request.to_role, request.top_n)
        elif request.method == "hybrid":
            from_recs = recommender.generate_hybrid_recommendations(request.from_role, request.top_n)
            to_recs = recommender.generate_hybrid_recommendations(request.to_role, request.top_n)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'content', 'collaborative', or 'hybrid'.")

        # Normalize skill sets
        from_skills = {rec['skill'].lower().strip(): rec for rec in from_recs}
        to_skills = {rec['skill'].lower().strip(): rec for rec in to_recs}

        # Compare sets
        shared_skills = [to_skills[s] for s in to_skills if s in from_skills]
        missing_skills = [to_skills[s] for s in to_skills if s not in from_skills]

        return {
            "from_role": request.from_role,
            "to_role": request.to_role,
            "shared_skills": shared_skills,
            "missing_skills": missing_skills,
            "method": request.method
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    if not os.path.exists("JobRoleToSkillRecommendation.xlsx"):
        print("No dataset found: create or upload JobRoleToSkillRecommendation.xlsx before running.")
    print("Training model with sample data...")
    success = recommender.train_full_pipeline("JobRoleToSkillRecommendation.xlsx")
    if success:
        print("Model training successful. Starting API server...")
        uvicorn.run(app, host="0.0.0.0", port=5000)
    else:
        print("Model training failed. Please check your data file.")
