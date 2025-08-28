import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import torch
import os
import warnings
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import time

# Set page configuration FIRST
st.set_page_config(
    page_title="LyricLens",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import MCR system
from music_content_rating import MusicContentRatingSystem

# Import transformer components
TRANSFORMERS_AVAILABLE = True
_transformers_warning = None
try:
    from transformers import LongformerTokenizer, LongformerForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    _transformers_warning = "Transformers library not available. Running in demo mode."

# Filter NumPy-related warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('wordnet')
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# Show transformers warning after page config
if _transformers_warning:
    st.warning(_transformers_warning)

# Apply conference-ready compact theme
st.markdown("""
<style>
    :root {
        --bg-root: #121212; --bg-card: #181818; --bg-alt: #1e1e1e; --accent: #1DB954; --accent-hover:#1ED760; --warn:#FF8C00; --danger:#E81123; --mid:#FFCD00; --text-primary:#FFFFFF; --text-secondary:#B3B3B3; --radius:12px; --transition:0.25s ease;
    }
    .main, .stApp { background-color: var(--bg-root); color: var(--text-primary); }
    body, p, span, div { font-family: 'Circular','Helvetica Neue',Arial,sans-serif; }
    h1 { font-size: 2.1rem; margin-bottom: .6rem; }
    h2,h3 { font-size: 1.25rem; letter-spacing:.5px; }
    .header { font-weight:600; }
    .compact-card { background:var(--bg-card); padding:20px 24px; border-radius:var(--radius); box-shadow:0 3px 8px rgba(0,0,0,.5); margin-bottom:20px; }
    .compact-card.tight { padding:12px 16px; }
    .flex { display:flex; }
    .gap-s { gap:10px; }
    .gap-m { gap:16px; }
    .center-y { align-items:center; }
    .justify-between { justify-content:space-between; }
    .badge { display:inline-flex; align-items:center; padding:3px 12px; font-size:.7rem; letter-spacing:.7px; border-radius:20px; font-weight:600; background:#222; text-transform:uppercase; }
    .badge.safe { background:rgba(29,185,84,0.15); color:var(--accent); }
    .badge.unsafe { background:rgba(232,17,35,0.15); color:var(--danger); }
    .mcr-hero { background:linear-gradient(135deg,#1f1f1f 0%,#181818 60%); position:relative; overflow:hidden; }
    .mcr-hero:before { content:""; position:absolute; inset:0; background:radial-gradient(circle at 85% 15%,rgba(29,185,84,.15),transparent 60%); }
    .mcr-rating-pill { font-size:2.2rem; font-weight:700; line-height:1; margin:0 0 8px; }
    .mcr-rating-pill.m-e { color:var(--accent); }
    .mcr-rating-pill.m-p { color:var(--mid); }
    .mcr-rating-pill.m-t { color:var(--warn); }
    .mcr-rating-pill.m-r { color:var(--danger); }
    .mcr-rating-pill.m-ao { color:#8B0000; }
    .descriptor-line { font-size:.9rem; color:var(--text-secondary); margin-bottom:6px; }
    .recommendation { font-size:.8rem; font-style:italic; color:#ccc; }
    .severity-bar { position:relative; height:6px; background:linear-gradient(90deg,var(--accent),var(--mid),var(--danger)); border-radius:4px; margin:12px 0 6px; }
    .severity-pointer { position:absolute; top:-4px; width:14px; height:14px; border-radius:50%; background:#fff; box-shadow:0 0 6px rgba(0,0,0,.6); }
    .severity-label { font-size:.7rem; text-transform:uppercase; letter-spacing:.5px; color:var(--text-secondary); }
    .mini-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
    .mini-metric { background:#202020; padding:8px 8px 6px; border-radius:6px; position:relative; }
    .mini-metric h4 { font-size:.55rem; font-weight:600; letter-spacing:.6px; margin:0 0 3px; text-transform:uppercase; color:var(--text-secondary); }
    .metric-value { font-size:.7rem; font-weight:600; }
    .bar-track { height:4px; background:#333; border-radius:3px; overflow:hidden; margin-top:3px; }
    .bar-fill { height:100%; background:var(--accent); transition:width var(--transition); }
    .bar-fill.mid { background:var(--warn); }
    .bar-fill.high { background:var(--danger); }
    .inline-meta { font-size:.65rem; color:var(--text-secondary); letter-spacing:.5px; }
    .stButton>button { background:var(--accent)!important; color:#000!important; font-weight:600!important; border-radius:32px!important; padding:.5rem 1.4rem!important; border:none!important; transition:var(--transition)!important; }
    .stButton>button:hover { background:var(--accent-hover)!important; transform:translateY(-2px)!important; }
    textarea { font-size:.9rem!important; }
    .footer-line { text-align:center; font-size:.65rem; color:var(--text-secondary); margin-top:32px; opacity:.7; }
    .adv-expander .streamlit-expanderHeader { font-size:.8rem; letter-spacing:.5px; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

# Define MultiLabelXGBoostClassifier class for loading the XGBoost model
class MultiLabelXGBoostClassifier:
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
            
        return predictions
    
    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], len(self.models), 2))
        
        for i, model in enumerate(self.models):
            probabilities[:, i, :] = model.predict_proba(X)
            
        return probabilities

# Longformer preprocessing functions
class LongformerPreprocessor:
    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.contractions = {
            r"\bI'm\b": "I am", r"\byou're\b": "you are", r"\bhe's\b": "he is", r"\bshe's\b": "she is",
            r"\bit's\b": "it is", r"\bwe're\b": "we are", r"\bthey're\b": "they are",
            r"\bI've\b": "I have", r"\byou've\b": "you have", r"\bwe've\b": "we have", r"\bthey've\b": "they have",
            r"\bI'll\b": "I will", r"\byou'll\b": "you will", r"\bhe'll\b": "he will", r"\bshe'll\b": "she will",
            r"\bwe'll\b": "we will", r"\bthey'll\b": "they will",
            r"\bI'd\b": "I would", r"\byou'd\b": "you would", r"\bhe'd\b": "he would", r"\bshe'd\b": "she would",
            r"\bwe'd\b": "we would", r"\bthey'd\b": "they would",
            r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
            r"\bwon't\b": "will not", r"\bwouldn't\b": "would not", r"\bcan't\b": "can not",
            r"\bcouldn't\b": "could not", r"\bshouldn't\b": "should not", r"\bmustn't\b": "must not",
            r"\bwasn't\b": "was not", r"\bweren't\b": "were not", r"\baren't\b": "are not", r"\bisn't\b": "is not",
            r"\bain't\b": "am not", r"\blet's\b": "let us"
        }
    
    def wn_pos(self, t):
        return {"J": "a", "V": "v", "N": "n", "R": "r"}.get(t[0], "n")
    
    def expand_contractions_str(self, text):
        for pattern, replacement in self.contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def clean_and_lemmatize(self, text: str) -> str:
        if pd.isna(text):
            return ""
        cleaned = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = self.expand_contractions_str(line)
            tokens = [w for w in nltk.word_tokenize(line.lower()) if w.isalpha()]
            if not tokens:
                continue
            tagged = nltk.pos_tag(tokens)
            lemmas = [self.lemmatizer.lemmatize(w, self.wn_pos(t)) for w, t in tagged]
            if lemmas:
                cleaned.append(" ".join(lemmas))
        merged = " ".join(cleaned)
        merged = re.sub(r"[^a-zA-Z\s]", "", merged)
        return re.sub(r"\s+", " ", merged).strip()

# Load models
@st.cache_resource
def load_xgboost_models():
    try:
        # Try loading enhanced models first
        enhanced_vectorizer_path = 'enhanced_tfidf_vectorizer.pkl'
        enhanced_model_path = 'enhanced_xgboost_model.pkl'
        
        # Fallback to original models
        original_vectorizer_path = 'safe_tfidf_vectorizer.pkl'
        original_model_path = 'safe_xgboost_model.pkl'
        
        # Try enhanced models first
        if os.path.exists(enhanced_vectorizer_path) and os.path.exists(enhanced_model_path):
            vectorizer_path = enhanced_vectorizer_path
            model_path = enhanced_model_path
            model_type = "Enhanced"
        elif os.path.exists(original_vectorizer_path) and os.path.exists(original_model_path):
            vectorizer_path = original_vectorizer_path
            model_path = original_model_path
            model_type = "Standard"
        else:
            # Silent failure - no error message to user
            return None, None
        
        # Load vectorizer with validation
        vectorizer = joblib.load(vectorizer_path)
        
        # Validate vectorizer is properly fitted - silent failure if not
        if not hasattr(vectorizer, 'idf_') or vectorizer.idf_ is None:
            return None, None
        
        if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
            return None, None
        
        # Load model with validation
        model = joblib.load(model_path)
        
        # Validate model - silent failure if not properly structured
        if not hasattr(model, 'models') or not model.models:
            return None, None
        
        # Silent loading for background use
        return vectorizer, model
        
    except Exception as e:
        # Silent failure - XGBoost not shown in UI
        return None, None

@st.cache_resource
def load_longformer_model():
    try:
        if not TRANSFORMERS_AVAILABLE:
            return None, None, None
        
        # Update these paths to your actual file locations
        # Option 1: Use forward slashes (works on Windows too)
        base_path = "C:/Users/malha/Desktop/kids-safe-music-INTEGRATED/checkpoint-3588"
        
        # Option 2: Use raw strings (uncomment if you prefer backslashes)
        # base_path = r"C:\Users\malha\Desktop\kids-safe-music-LATEST\checkpoint-3588"
        
        # Option 3: Use os.path.join (most robust)
        # import os
        # base_path = os.path.join("C:", "Users", "malha", "Desktop", "kids-safe-music-LATEST", "checkpoint-3588")
        
        model_path = f"{base_path}/model.safetensors"  # Path to your safetensors file
        config_path = f"{base_path}/config.json"      # Path to your config file
        tokenizer_path = base_path                     # Directory containing tokenizer files
        
        # Check if required files exist
        if not os.path.exists(model_path):
            st.warning(f"Model file not found: {model_path}")
            return None, None, None
        
        if not os.path.exists(config_path):
            st.warning(f"Config file not found: {config_path}")
            return None, None, None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer (should work with directory containing tokenizer files)
        tokenizer = LongformerTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        
        # Load model using safetensors
        from safetensors.torch import load_file
        
        # Load the model configuration
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(config_path, local_files_only=True)
        
        # Create model with config
        model = LongformerForSequenceClassification(config)
        
        # Load weights from safetensors file
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model = model.to(device).eval()
        
        preprocessor = LongformerPreprocessor()
        
        return tokenizer, model, preprocessor
        
    except ImportError as e:
        st.error("safetensors library not found. Please install it: pip install safetensors")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading Longformer model: {str(e)}")
        st.error("Make sure you have: model.safetensors, config.json, and tokenizer files")
        return None, None, None

# XGBoost preprocessing functions (your existing ones)
def preprocess_lyrics_xgboost(lyrics):
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    tokens = word_tokenize(lyrics)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    preprocessed_lyrics = ' '.join(filtered_tokens)
    return preprocessed_lyrics

def create_windows(text, window_size=10):
    words = text.split()
    windows = []
    
    if len(words) <= window_size:
        return [text]
    
    for i in range(len(words) - window_size + 1):
        window = ' '.join(words[i:i+window_size])
        windows.append(window)
    
    return windows

def post_process_violence(windows, predictions):
    violence_terms = [
        'kill', 'murder', 'shot', 'gun', 'shoot', 'stab', 'punch', 'kick', 'beat', 
        'hit', 'fight', 'hurt', 'attack', 'blood', 'dead', 'death', 'die', 'knife',
        'weapon', 'wound', 'bleed', 'brutal', 'slap', 'choke', 'strangle'
    ]
    
    for i, window in enumerate(windows):
        window_lower = window.lower()
        has_violence_term = any(term in window_lower for term in violence_terms)
        
        if predictions[i, 1] == 1 and not has_violence_term:
            predictions[i, 1] = 0
    
    return predictions

# XGBoost analysis function
def analyze_lyrics_xgboost(lyrics, model, vectorizer):
    try:
        # Validate inputs
        if not hasattr(vectorizer, 'transform'):
            raise ValueError("Vectorizer is not properly loaded")
        
        if not hasattr(vectorizer, 'idf_') or vectorizer.idf_ is None:
            raise ValueError("Vectorizer is not fitted - missing IDF values")
        
        cleaned_lyrics = preprocess_lyrics_xgboost(lyrics)
        
        if not cleaned_lyrics or len(cleaned_lyrics.strip()) == 0:
            st.warning("Lyrics became empty after preprocessing")
            total_words = len(lyrics.split())
            return {
                'explicit': 0,
                'violence': 0,
                'sexual_content': 0,
                'substance_use': 0,
                'total_words': total_words,
                'windows': [],
                'window_predictions': np.array([])
            }
        
        total_words = len(cleaned_lyrics.split())
        
        if total_words == 0:
            st.warning("No valid words found after preprocessing")
            return {
                'explicit': 0,
                'violence': 0,
                'sexual_content': 0,
                'substance_use': 0,
                'total_words': len(lyrics.split()),
                'windows': [],
                'window_predictions': np.array([])
            }
        
        explicit_words = 0
        violence_words = 0
        sexual_words = 0
        substance_words = 0
        
        windows = create_windows(cleaned_lyrics)
        
        if not windows:
            st.warning("No windows created from lyrics")
            return {
                'explicit': 0,
                'violence': 0,
                'sexual_content': 0,
                'substance_use': 0,
                'total_words': total_words,
                'windows': [],
                'window_predictions': np.array([])
            }
        
        # Transform windows with error handling
        try:
            X_windows = vectorizer.transform(windows)
        except Exception as e:
            st.error(f"Error in vectorizer.transform: {str(e)}")
            raise ValueError(f"Vectorizer transform failed: {str(e)}")
        
        # Make predictions with error handling
        try:
            window_predictions = model.predict(X_windows)
        except Exception as e:
            st.error(f"Error in model.predict: {str(e)}")
            raise ValueError(f"Model prediction failed: {str(e)}")
        
        window_predictions = post_process_violence(windows, window_predictions)
        
        # Map words to windows (your existing logic)
        word_to_windows = {}
        for i, word in enumerate(cleaned_lyrics.split()):
            word_to_windows[i] = []
            
        for i, window in enumerate(windows):
            window_words = window.split()
            word_indices = []
            
            words = cleaned_lyrics.split()
            for j in range(len(words) - len(window_words) + 1):
                if ' '.join(words[j:j+len(window_words)]) == window:
                    word_indices = list(range(j, j+len(window_words)))
                    break
                    
            for idx in word_indices:
                if idx in word_to_windows:
                    if len(window_predictions[i]) > 0 and window_predictions[i, 0]:
                        word_to_windows[idx].append('explicit')
                    if len(window_predictions[i]) > 1 and window_predictions[i, 1]:
                        word_to_windows[idx].append('violence')
                    if len(window_predictions[i]) > 2 and window_predictions[i, 2]:
                        word_to_windows[idx].append('sexual')
                    if len(window_predictions[i]) > 3 and window_predictions[i, 3]:
                        word_to_windows[idx].append('substance')
        
        for idx, categories in word_to_windows.items():
            if 'explicit' in categories:
                explicit_words += 1
            if 'violence' in categories:
                violence_words += 1
            if 'sexual' in categories:
                sexual_words += 1
            if 'substance' in categories:
                substance_words += 1
        
        results = {
            'explicit': min(100, (explicit_words / total_words) * 100) if total_words > 0 else 0,
            'violence': min(100, (violence_words / total_words) * 100) if total_words > 0 else 0,
            'sexual_content': min(100, (sexual_words / total_words) * 100) if total_words > 0 else 0,
            'substance_use': min(100, (substance_words / total_words) * 100) if total_words > 0 else 0,
            'window_predictions': window_predictions,
            'windows': windows,
            'total_words': total_words
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error in XGBoost analysis: {str(e)}")
        raise e

# Longformer analysis function
def analyze_lyrics_longformer(lyrics, tokenizer, model, preprocessor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cleaned = preprocessor.clean_and_lemmatize(lyrics)
    if not cleaned:
        return {
            'explicit': 0,
            'violence': 0,
            'sexual_content': 0,
            'substance_use': 0,
            'total_words': len(lyrics.split()),
            'confidence_scores': [0, 0, 0, 0],
            'raw_probabilities': [0, 0, 0, 0]
        }
    
    enc = tokenizer(cleaned, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
    enc["global_attention_mask"] = torch.zeros_like(enc["input_ids"])
    enc["global_attention_mask"][:, 0] = 1
    
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in enc.items()}
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Map Longformer labels to our format
    # Longformer labels: ["sexual", "violence", "substance", "language"]
    # Our format: ["explicit", "violence", "sexual_content", "substance_use"]
    
    # Category-specific probability scaling for better accuracy
    def category_specific_scaling(prob, category):
        """
        Use different thresholds and scaling approaches for different categories
        Language model appears severely miscalibrated - using step function approach
        """
        if category == 'explicit':
            # Step function approach for language - only flag truly extreme content
            if prob < 0.95:  # Require 95%+ confidence for any explicit rating
                return 0.0
            elif prob < 0.98:  # 95-98% maps to low explicit (0-30%)
                return ((prob - 0.95) / 0.03) * 30
            else:  # 98%+ maps to high explicit (30-100%)
                return 30 + ((prob - 0.98) / 0.02) * 70
                
        elif category == 'sexual':
            # High threshold but gradual scaling for sexual content
            if prob < 0.75:
                return 0.0
            else:
                return ((prob - 0.75) / 0.25) * 100
                
        else:  # violence and substance
            # Standard scaling for these categories
            threshold = 0.35
            if prob < threshold:
                return 0.0
            else:
                return ((prob - threshold) / (1.0 - threshold)) * 100
    
    # Use category-specific scaling approach
    results = {
        'explicit': category_specific_scaling(probs[3], 'explicit'),      # language -> explicit
        'violence': category_specific_scaling(probs[1], 'violence'),      # violence -> violence  
        'sexual_content': category_specific_scaling(probs[0], 'sexual'),  # sexual -> sexual_content
        'substance_use': category_specific_scaling(probs[2], 'substance'), # substance -> substance_use
        'total_words': len(lyrics.split()),
        'confidence_scores': probs.tolist(),
        'raw_probabilities': probs.tolist(),
        'scaling_method': 'category_specific'
    }
    
    # Ensure no negative values and reasonable bounds
    for key in ['explicit', 'violence', 'sexual_content', 'substance_use']:
        results[key] = max(0.0, min(100.0, results[key]))
    
    return results

# Weighted Severity Index
def calculate_content_severity(results):
    weights = {
        'explicit': 0.25,
        'violence': 0.30,
        'sexual_content': 0.25,
        'substance_use': 0.20
    }
    
    severity = sum(results[category] * weights[category] for category in weights)
    return min(100, severity)

# Radar Chart stub (removed for conference)
def create_radar_chart(results):
    # Charts removed for conference demo
    return None

# Classification function
def classify_lyrics(lyrics, model_type, **kwargs):
    if model_type == "XGBoost":
        vectorizer = kwargs['vectorizer']
        model = kwargs['model']
        results = analyze_lyrics_xgboost(lyrics, model, vectorizer)
    else:  # Longformer
        tokenizer = kwargs['tokenizer']
        model = kwargs['model']
        preprocessor = kwargs['preprocessor']
        results = analyze_lyrics_longformer(lyrics, tokenizer, model, preprocessor)
    
    # Calculate severity
    severity = calculate_content_severity(results)
    results['severity'] = severity
    
    # Calculate MCR Rating
    mcr_system = MusicContentRatingSystem()
    mcr_result = mcr_system.calculate_rating(
        violence_score=results['violence'] / 100.0,  # Convert percentage to 0-1 scale
        sexual_score=results['sexual_content'] / 100.0,
        language_score=results['explicit'] / 100.0,
        substance_score=results['substance_use'] / 100.0
    )
    results['mcr_rating'] = mcr_result['rating']
    results['mcr_descriptors'] = mcr_result['descriptors']
    results['mcr_recommendation'] = mcr_result['details']['recommendation']
    results['mcr_explanation'] = mcr_result['details']['rating_explanation']
    
    # Determine safety levels
    threshold_high = 30
    threshold_medium = 10
    violence_threshold_medium = 15
    violence_threshold_high = 35
    
    if (results['explicit'] > threshold_medium or
        results['violence'] > violence_threshold_medium or
        results['sexual_content'] > threshold_medium or 
        results['substance_use'] > threshold_medium):
        results['kid_safe'] = False
    else:
        results['kid_safe'] = True
    
    if (results['explicit'] > threshold_high or
        results['violence'] > violence_threshold_high or
        results['sexual_content'] > threshold_high or 
        results['substance_use'] > threshold_high):
        results['safety_level'] = "Explicit"
        results['rating_color'] = "#E81123"
    elif (results['explicit'] > threshold_medium or
          results['violence'] > violence_threshold_medium or
          results['sexual_content'] > threshold_medium or 
          results['substance_use'] > threshold_medium):
        results['safety_level'] = "Moderate"
        results['rating_color'] = "#FF8C00"
    else:
        results['safety_level'] = "Clean"
        results['rating_color'] = "#1DB954"
    
    return results

# Display results function (redesigned for conference)
def display_results(results):
    kid_safe = results.get('kid_safe', True)
    safety_level = results.get('safety_level','Clean')
    safety_class = 'safe' if kid_safe else 'unsafe'

    mcr_rating = results.get('mcr_rating','M-E')
    mcr_descriptors = results.get('mcr_descriptors', [])
    descriptors_text = '; '.join(mcr_descriptors) if mcr_descriptors else ('Everyone' if mcr_rating=='M-E' else 'None')
    mcr_recommendation = results.get('mcr_recommendation','')
    mcr_explanation = results.get('mcr_explanation','')
    mcr_class = mcr_rating.lower().replace('-', '-')

    # Filter out "Adults only" and similar recommendations
    if mcr_recommendation and ('adults only' in mcr_recommendation.lower() or 'not suitable for minors' in mcr_recommendation.lower()):
        mcr_recommendation = ''

    severity = results.get('severity',0)

    # Hero MCR rating card
    # Add detailed rating descriptions
    rating_descriptions = {
        'M-E': 'Everyone',
        'M-P': 'Parental Guidance Suggested',
        'M-T': 'Teen',
        'M-R': 'Restricted',
        'M-AO': 'Adults Only'
    }
    
    rating_detail = rating_descriptions.get(mcr_rating, '')
    
    st.markdown(f"""
    <div class="compact-card mcr-hero">
        <div class="flex justify-between center-y">
            <div>
                <div class="mcr-rating-pill {mcr_class}" aria-label="Music Content Rating">{mcr_rating}</div>
                <div class="descriptor-line"><strong>{rating_detail}</strong></div>
                <div class="descriptor-line"><strong>Content:</strong> {descriptors_text}</div>
                <div class="recommendation">{mcr_recommendation}</div>
            </div>
            <div style="text-align:right;">
                <span class="badge {safety_class}" aria-label="Kid Safety Status">{ 'SAFE' if kid_safe else 'UNSAFE' }</span><br>
                <span class="inline-meta" style="color:{results.get('rating_color','#1DB954')}">{safety_level}</span>
            </div>
        </div>
        <div class="severity-bar" aria-label="Severity Index">
            <div class="severity-pointer" style="left:{severity}%;"></div>
        </div>
        <div class="flex justify-between" style="margin-top:2px;">
            <span class="severity-label">LOW</span>
            <span style="font-size:.65rem; font-weight:600;">SEVERITY {severity:.1f}</span>
            <span class="severity-label">HIGH</span>
        </div>
        <div style="margin-top:6px; font-size:.6rem; color:#888;">{mcr_explanation}</div>
    </div>
    """, unsafe_allow_html=True)

    # Content breakdown mini grid
    explicit = results.get('explicit',0)
    violence = results.get('violence',0)
    sexual = results.get('sexual_content',0)
    substance = results.get('substance_use',0)

    def bar_class(v, high=30, mid=10, special_high=None, special_mid=None):
        h = special_high if special_high is not None else high
        m = special_mid if special_mid is not None else mid
        if v > h: return 'high'
        if v > m: return 'mid'
        return ''

    st.markdown(f"""
    <div class="compact-card tight">
        <div style="font-size:.65rem; font-weight:600; letter-spacing:.5px; margin-bottom:4px; color:var(--text-secondary); text-transform:uppercase;">Content Breakdown</div>
        <div class="mini-grid">
            <div class="mini-metric">
                <h4>Explicit</h4>
                <div class="metric-value">{explicit:.1f}%</div>
                <div class="bar-track"><div class="bar-fill {bar_class(explicit)}" style="width:{explicit}%;"></div></div>
            </div>
            <div class="mini-metric">
                <h4>Violence</h4>
                <div class="metric-value">{violence:.1f}%</div>
                <div class="bar-track"><div class="bar-fill {bar_class(violence, high=35, mid=15)}" style="width:{violence}%;"></div></div>
            </div>
            <div class="mini-metric">
                <h4>Sexual</h4>
                <div class="metric-value">{sexual:.1f}%</div>
                <div class="bar-track"><div class="bar-fill {bar_class(sexual)}" style="width:{sexual}%;"></div></div>
            </div>
            <div class="mini-metric">
                <h4>Substance</h4>
                <div class="metric-value">{substance:.1f}%</div>
                <div class="bar-track"><div class="bar-fill {bar_class(substance)}" style="width:{substance}%;"></div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Advanced details expander
    with st.expander("Advanced Details", expanded=False):
        raw_probs = results.get('raw_probabilities', [])
        if raw_probs:
            st.markdown("**Raw Model Probabilities (Longformer)**")
            st.write({ 'sexual': raw_probs[0], 'violence': raw_probs[1], 'substance': raw_probs[2], 'language/explicit': raw_probs[3] })
        st.caption("Charts removed for conference demo version. Window analysis available only for XGBoost (currently not active in UI).")

# Main application
def main():
    # Conference-ready header
    st.markdown("""
    <div class="flex center-y justify-between" style="margin-bottom:18px;">
        <div>
            <h1 style="margin:0;">LyricLens</h1>
            <div style="font-size:.8rem; color:var(--text-secondary);">AI-Powered Lyric Content Assessment</div>
        </div>
        <div style="text-align:right; font-size:.55rem; color:var(--text-secondary); letter-spacing:.5px;">Conference Demo • 2025 Edition</div>
    </div>
    """, unsafe_allow_html=True)

    # Load models (silent for XGBoost)
    _xgb_vectorizer, _xgb_model = load_xgboost_models()  # kept but unused
    longformer_tokenizer, longformer_model, longformer_preprocessor = load_longformer_model()

    if not all([longformer_tokenizer, longformer_model, longformer_preprocessor]):
        st.error("Longformer model unavailable. Running in simulated demo mode.")

    col_input, col_results = st.columns([5,7], gap="large")

    with col_input:
        st.markdown("<div class='compact-card'><div style='font-size:.8rem; font-weight:600; letter-spacing:.5px; color:var(--text-secondary); text-transform:uppercase; margin-bottom:6px;'>Input Lyrics</div>", unsafe_allow_html=True)
        lyrics = st.text_area("Paste lyrics here", height=260, placeholder="Enter song lyrics to analyze...", label_visibility="collapsed")
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            if lyrics:
                st.caption(f"Words: {len(lyrics.split())}")
        with meta_col2:
            if lyrics:
                st.caption(f"Chars: {len(lyrics)}")
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        result_placeholder = st.container()

    if analyze_clicked:
        if not lyrics.strip():
            st.warning("Please provide lyrics before analysis.")
        else:
            with st.spinner("Analyzing with transformer model..."):
                try:
                    if all([longformer_tokenizer, longformer_model, longformer_preprocessor]):
                        results = classify_lyrics(
                            lyrics, "Longformer",
                            tokenizer=longformer_tokenizer,
                            model=longformer_model,
                            preprocessor=longformer_preprocessor
                        )
                    else:
                        # Demo fallback (simulated)
                        demo_mcr_ratings = ['M-E','M-P','M-T','M-R','M-AO']
                        demo_mcr_rating = random.choice(demo_mcr_ratings)
                        demo_desc_map = {
                            'M-E': [], 'M-P': ['Mild Language'], 'M-T': ['Strong Language','Suggestive Themes'], 'M-R': ['Explicit Language','Violence'], 'M-AO': ['Explicit Language','Graphic Violence','Sexual Content']
                        }
                        results = {
                            'explicit': random.uniform(0,40), 'violence': random.uniform(0,35), 'sexual_content': random.uniform(0,30), 'substance_use': random.uniform(0,25)
                        }
                        results['severity'] = calculate_content_severity(results)
                        results['kid_safe'] = all(v < 10 for k,v in results.items() if k in ['explicit','violence','sexual_content','substance_use'])
                        results['safety_level'] = 'Clean' if results['kid_safe'] else 'Moderate'
                        results['rating_color'] = '#1DB954' if results['kid_safe'] else '#FF8C00'
                        mcr_system = MusicContentRatingSystem()
                        mcr_res = mcr_system.calculate_rating(
                            violence_score=results['violence']/100.0,
                            sexual_score=results['sexual_content']/100.0,
                            language_score=results['explicit']/100.0,
                            substance_score=results['substance_use']/100.0
                        )
                        results['mcr_rating'] = mcr_res['rating']
                        results['mcr_descriptors'] = mcr_res['descriptors']
                        results['mcr_recommendation'] = mcr_res['details']['recommendation'] + ' (Simulated)'
                        results['mcr_explanation'] = mcr_res['details']['rating_explanation']
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return
            with result_placeholder:
                display_results(results)

    st.markdown("<div class='footer-line'>© 2025 LyricLens • Advanced music content analysis platform</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()