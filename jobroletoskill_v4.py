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

        if 'Job Role' not in self.data.columns or 'Skill Title' not in self.data.columns:
            raise ValueError("Expected columns 'Job Role' and 'Skill Title' in input data.")

        self.data['Job Role'] = self.data['Job Role'].astype(str).str.lower().str.strip()
        self.data['Skill Title'] = self.data['Skill Title'].astype(str).str.lower().str.strip()

        self.job_roles = self.data['Job Role'].unique().tolist()
        self.skills = self.data['Skill Title'].unique().tolist()
        print(f"Found {len(self.job_roles)} unique job roles and {len(self.skills)} unique skills.")

        categorical_cols = ['Industry', 'Department', 'Skill Category', 'Skill Type', 'Proficiency Level']
        available_cols = [col for col in categorical_cols if col in self.data.columns]
        for col in available_cols:
            encoder_name = f"{col.lower().replace(' ', '_')}_encoder"
            if hasattr(self, encoder_name):
                encoder = getattr(self, encoder_name)
                self.data[f"{col}_encoded"] = encoder.fit_transform(self.data[col].astype(str))

        def create_job_role_text(row):
            parts = []
            job_role_clean = self._clean_text(self._process_field(row['Job Role']))
            if job_role_clean: parts.append(job_role_clean)
            if 'Industry' in row and pd.notna(row['Industry']) and str(row['Industry']).strip():
                parts.append(f"in {self._clean_text(str(row['Industry']))} industry")
            if 'Department' in row and pd.notna(row['Department']) and str(row['Department']).strip():
                parts.append(f"{self._clean_text(str(row['Department']))} department")
            return " ".join(parts)

        def create_skill_text(row):
            parts = []
            skill_clean = self._clean_text(self._process_field(row['Skill Title']))
            if skill_clean: parts.append(skill_clean)
            for field, desc in [('Skill Category', 'category'), ('Skill Type', 'type'), ('Proficiency Level', 'proficiency level')]:
                if field in row and pd.notna(row[field]) and str(row[field]).strip():
                    parts.append(f"{desc}: {self._clean_text(str(row[field]))}")
            return ". ".join(dict.fromkeys(parts))

        self.data['Job_Role_Text'] = self.data.apply(create_job_role_text, axis=1)
        self.data['Skill_Text'] = self.data.apply(create_skill_text, axis=1)
        print("Enhanced feature engineering completed.")
        return True

    # ---------------------------
    # Embeddings
    # ---------------------------
    def create_embeddings(self):
        job_role_data = self.data[['Job Role', 'Job_Role_Text']].drop_duplicates()
        skill_data = self.data[['Skill Title', 'Skill_Text']].drop_duplicates()
        valid_job_roles, job_role_texts = [], []
        seen_roles = set()
        for _, row in job_role_data.iterrows():
            role = row['Job Role']
            if role not in seen_roles and pd.notna(row['Job_Role_Text']) and row['Job_Role_Text'].strip():
                valid_job_roles.append(role)
                job_role_texts.append(row['Job_Role_Text'])
                seen_roles.add(role)
                valid_skills, skill_texts = [], []
        seen_skills = set()
        for _, row in skill_data.iterrows():
            skill = row['Skill Title']
            if skill not in seen_skills and pd.notna(row['Skill_Text']) and row['Skill_Text'].strip():
                valid_skills.append(skill)
                skill_texts.append(row['Skill_Text'])
                seen_skills.add(skill)
        self.job_role_embeddings = self.embedding_model.encode(job_role_texts, show_progress_bar=True, batch_size=32, convert_to_tensor=True)
        self.skill_embeddings = self.embedding_model.encode(skill_texts, show_progress_bar=True, batch_size=32, convert_to_tensor=True)
        self.job_role_to_idx = {role: idx for idx, role in enumerate(valid_job_roles)}
        self.idx_to_job_role = {idx: role for role, idx in self.job_role_to_idx.items()}
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(valid_skills)}
        self.idx_to_skill = {idx: skill for skill, idx in self.skill_to_idx.items()}
        print(f"Embeddings created. Job roles: {len(job_role_texts)}, Skills: {len(skill_texts)}")
        return True

    # ---------------------------
    # Content similarity
    # ---------------------------
    def compute_content_similarity(self):
        from sklearn.preprocessing import normalize
        job_embeddings = self.job_role_embeddings.numpy() if hasattr(self.job_role_embeddings, 'numpy') else self.job_role_embeddings
        skill_embeddings = self.skill_embeddings.numpy() if hasattr(self.skill_embeddings, 'numpy') else self.skill_embeddings
        job_embeddings_norm = normalize(job_embeddings)
        skill_embeddings_norm = normalize(skill_embeddings)
        self.content_similarity_matrix = cosine_similarity(job_embeddings_norm, skill_embeddings_norm)
        self.similarity_stats = {'mean': np.mean(self.content_similarity_matrix),
                                 'std': np.std(self.content_similarity_matrix),
                                 'max': np.max(self.content_similarity_matrix),
                                 'min': np.min(self.content_similarity_matrix)}
        return True

    # ---------------------------
    # Collaborative data & model
    # ---------------------------
    def prepare_collaborative_data(self):
        interactions = self.data.groupby(['Job Role', 'Skill Title']).size().reset_index(name='count')

        # keep only job roles and skills that exist in our dictionaries
        interactions = interactions[
            interactions['Job Role'].isin(self.job_role_to_idx.keys()) &
            interactions['Skill Title'].isin(self.skill_to_idx.keys())
        ]

        interactions['role_idx'] = interactions['Job Role'].map(self.job_role_to_idx)
        interactions['skill_idx'] = interactions['Skill Title'].map(self.skill_to_idx)

        n_roles, n_skills = len(self.job_role_to_idx), len(self.skill_to_idx)
        self.interaction_matrix = csr_matrix(
            (interactions['count'], (interactions['role_idx'], interactions['skill_idx'])),
            shape=(n_roles, n_skills)
        )
        return True


    def train_collaborative_model(self, n_components=50):
        if self.interaction_matrix is None:
            return False
        self.collab_model = NMF(
            n_components=n_components,
            init='nndsvd',
            random_state=42,
            max_iter=500,
            alpha_W=0.1,
            alpha_H=0.1,
            l1_ratio=0.5
        )
        W = self.collab_model.fit_transform(self.interaction_matrix)
        H = self.collab_model.components_
        self.collab_similarity_matrix = np.dot(W, H)
        return True


    # ---------------------------
    # Recommendations
    # ---------------------------
    def generate_content_recommendations(self, job_role, top_n=5):
        job_role = job_role.lower().strip()
        if job_role not in self.job_role_to_idx: raise ValueError(f"Job role '{job_role}' not found.")
        sims = self.content_similarity_matrix[self.job_role_to_idx[job_role]]
        top_idx = np.argsort(sims)[::-1][:top_n]
        return [{'skill': self.idx_to_skill[i], 'score': float(sims[i])} for i in top_idx]

    def generate_collaborative_recommendations(self, job_role, top_n=5):
        job_role = job_role.lower().strip()
        sims = self.collab_similarity_matrix[self.job_role_to_idx[job_role]]
        top_idx = np.argsort(sims)[::-1][:top_n]
        return [{'skill': self.idx_to_skill[i], 'score': float(sims[i])} for i in top_idx]

    def generate_hybrid_recommendations(self, job_role, top_n=5, content_weight=0.5, collab_weight=0.5):
        job_role = job_role.lower().strip()
        content_scores = self.content_similarity_matrix[self.job_role_to_idx[job_role]]
        collab_scores = self.collab_similarity_matrix[self.job_role_to_idx[job_role]]
        content_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
        collab_norm = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min())
        hybrid = content_weight * content_norm + collab_weight * collab_norm
        top_idx = np.argsort(hybrid)[::-1][:top_n]
        return [{'skill': self.idx_to_skill[i], 'score': float(hybrid[i])} for i in top_idx]

    # ---------------------------
    # Training pipeline & persistence
    # ---------------------------
    def train_full_pipeline(self, file_path):
        self.ingest_data(file_path)
        self.preprocess_data()
        self.feature_engineering()
        self.create_embeddings()
        self.compute_content_similarity()
        self.prepare_collaborative_data()
        self.train_collaborative_model()
        self.is_trained = True
        return True

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


# ---------------------------
# Persistence helpers
# ---------------------------
recommender = JobRoleSkillRecommender()
MODEL_PATH = "recommender.pkl"
DATA_PATH = "JobRoleToSkillRecommendation.xlsx"
_last_model_timestamp = None

def load_or_train_model():
    import os
    global recommender, _last_model_timestamp
    if os.path.exists(MODEL_PATH):
        recommender = JobRoleSkillRecommender.load_model(MODEL_PATH)
        recommender.is_trained = True
        _last_model_timestamp = os.path.getmtime(MODEL_PATH)
        return True
    elif os.path.exists(DATA_PATH):
        recommender.train_full_pipeline(DATA_PATH)
        recommender.save_model(MODEL_PATH)
        _last_model_timestamp = os.path.getmtime(MODEL_PATH)
        return True
    return False

def load_latest_model():
    import os
    global recommender, _last_model_timestamp
    if os.path.exists(MODEL_PATH):
        ts = os.path.getmtime(MODEL_PATH)
        if _last_model_timestamp is None or ts > _last_model_timestamp:
            recommender = JobRoleSkillRecommender.load_model(MODEL_PATH)
            recommender.is_trained = True
            _last_model_timestamp = ts
            print("Reloaded updated recommender from pickle.")


# ---------------------------
# FastAPI endpoints
# ---------------------------
@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    recommender.train_full_pipeline(file_path)
    recommender.save_model(MODEL_PATH)
    return {"message": "Data uploaded, model retrained & saved"}

@app.get("/recommendations")
async def get_recommendations(job_role: str, top_n: int = 5, method: str = "hybrid"):
    load_latest_model()
    if method == "content":
        recs = recommender.generate_content_recommendations(job_role, top_n)
    elif method == "collaborative":
        recs = recommender.generate_collaborative_recommendations(job_role, top_n)
    elif method == "hybrid":
        recs = recommender.generate_hybrid_recommendations(job_role, top_n)
    else:
        raise HTTPException(status_code=400, detail="Invalid method")
    return {"job_role": job_role, "recommendations": recs, "method": method}

@app.post("/retrain/")
async def retrain_model():
    load_latest_model()
    if recommender.data is None:
        raise HTTPException(status_code=400, detail="No data available")
    recommender.train_full_pipeline(DATA_PATH)
    recommender.save_model(MODEL_PATH)
    return {"message": "Model retrained & saved"}
from pydantic import BaseModel

class SkillGapRequest(BaseModel):
    current_skills: List[str]
    target_role: str
    top_n: Optional[int] = 10

class RoleGapRequest(BaseModel):
    from_role: str
    to_role: str
    top_n: Optional[int] = 10
@app.post("/skill_gap")
async def skill_gap_analysis(request: SkillGapRequest):
    load_latest_model()
    recommendations = recommender.generate_hybrid_recommendations(request.target_role, request.top_n)
    current = {s.lower().strip() for s in request.current_skills}
    required = {r['skill'].lower().strip(): r for r in recommendations}
    missing = [r for s, r in required.items() if s not in current]
    matched = [r for s, r in required.items() if s in current]
    return {
        "target_role": request.target_role,
        "current_skills": list(current),
        "matched_skills": matched,
        "missing_skills": missing,
    }

@app.post("/role_gap")
async def role_gap_analysis(request: RoleGapRequest):
    load_latest_model()
    from_recs = recommender.generate_hybrid_recommendations(request.from_role, request.top_n)
    to_recs = recommender.generate_hybrid_recommendations(request.to_role, request.top_n)
    from_sk = {r['skill'].lower().strip(): r for r in from_recs}
    to_sk = {r['skill'].lower().strip(): r for r in to_recs}
    shared = [to_sk[s] for s in to_sk if s in from_sk]
    missing = [to_sk[s] for s in to_sk if s not in from_sk]
    return {
        "from_role": request.from_role,
        "to_role": request.to_role,
        "shared_skills": shared,
        "missing_skills": missing,
    }

@app.get("/health/")
async def health_check():
    return {"status": "healthy", "model_trained": recommender.is_trained}

if __name__ == "__main__":
    if load_or_train_model():
        uvicorn.run(app, host="0.0.0.0", port=5000)
    else:
        print("Model initialization failed. Please upload data.")
