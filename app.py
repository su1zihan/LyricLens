import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import torch
import os
import warnings
import random
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

st.set_page_config(
    page_title="LyricLens",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from music_content_rating import MusicContentRatingSystem

TRANSFORMERS_AVAILABLE = True
_transformers_warning = None
try:
    from transformers import LongformerTokenizer, LongformerForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    _transformers_warning = "Transformers library not available. Running in demo mode."

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Ensure NLTK data path is set correctly
import nltk
try:
    nltk.data.path.append(os.path.expanduser('~/nltk_data'))
except:
    pass

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)  # Additional wordnet data
    except Exception as e:
        st.warning(f"NLTK download warning: {e}. Some features may be limited.")

if _transformers_warning:
    st.warning(_transformers_warning)

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

class LongformerPreprocessor:
    def __init__(self):
        try:
            # Force reload NLTK data to avoid lazy loading issues
            import nltk.stem
            import nltk.corpus
            
            # Initialize the lemmatizer with explicit error handling
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            
            # Test the lemmatizer to ensure it works
            test_result = self.lemmatizer.lemmatize("testing", "v")
            if not isinstance(test_result, str):
                raise ValueError("Lemmatizer returned unexpected type")
                
        except Exception as e:
            st.warning(f"WordNet lemmatizer initialization failed: {e}. Using basic text processing.")
            self.lemmatizer = None
        
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
            try:
                tokens = [w for w in nltk.word_tokenize(line.lower()) if w.isalpha()]
                if not tokens:
                    continue
                    
                if self.lemmatizer:
                    # Use lemmatizer if available
                    tagged = nltk.pos_tag(tokens)
                    lemmas = [self.lemmatizer.lemmatize(w, self.wn_pos(t)) for w, t in tagged]
                    if lemmas:
                        cleaned.append(" ".join(lemmas))
                else:
                    # Fallback to simple tokenization without lemmatization
                    if tokens:
                        cleaned.append(" ".join(tokens))
            except Exception:
                # Fallback to basic word extraction if NLTK fails
                basic_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', line)]
                if basic_tokens:
                    cleaned.append(" ".join(basic_tokens))
                    
        merged = " ".join(cleaned)
        merged = re.sub(r"[^a-zA-Z\s]", "", merged)
        return re.sub(r"\s+", " ", merged).strip()

def parse_arguments():
    """Parse command-line arguments for model path configuration."""
    parser = argparse.ArgumentParser(description='LyricLens - AI-Powered Lyric Content Assessment')
    parser.add_argument(
        '--model-path', 
        type=str, 
        default=None,
        help='Path to the model directory (default: ./model, fallback: ./checkpoint-3588)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port for Streamlit server (default: 8501)'
    )
    parser.add_argument(
        '--host', 
        type=str, 
        default='localhost',
        help='Host for Streamlit server (default: localhost)'
    )
    
    # For Streamlit, we need to handle the case where sys.argv might contain Streamlit-specific args
    import sys
    
    # Filter out Streamlit arguments that might interfere
    filtered_argv = []
    skip_next = False
    
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
            
        # Skip Streamlit-specific arguments
        if arg.startswith('--server.') or arg.startswith('--global.') or arg.startswith('--logger.'):
            if '=' not in arg and i + 1 < len(sys.argv):
                skip_next = True
            continue
        
        # Skip the script name if it contains 'streamlit'
        if i == 0 and ('streamlit' in arg or 'run' in arg):
            continue
            
        filtered_argv.append(arg)
    
    # If no model path arguments found, return defaults
    if not any('--model-path' in arg for arg in filtered_argv):
        return argparse.Namespace(
            model_path=None,
            port=8501,
            host='localhost'
        )
    
    # Parse the filtered arguments
    args = parser.parse_args(filtered_argv[1:] if filtered_argv else [])
    return args

@st.cache_resource
def load_longformer_model():
    try:
        if not TRANSFORMERS_AVAILABLE:
            return None, None, None
        
        # Parse command-line arguments
        args = parse_arguments()
        
        # Determine the model path
        if args.model_path:
            # Use user-specified path
            base_path = os.path.abspath(args.model_path)
        else:
            # Smart model path discovery
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Try common model folder names in order of preference
            candidate_folders = ["model", "checkpoint-3588", "models", "checkpoint"]
            base_path = None
            
            for folder_name in candidate_folders:
                candidate_path = os.path.join(script_dir, folder_name)
                model_file = os.path.join(candidate_path, "model.safetensors")
                config_file = os.path.join(candidate_path, "config.json")
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    base_path = candidate_path
                    break
            
            # Default to 'model' folder if nothing found
            if base_path is None:
                base_path = os.path.join(script_dir, "model")
        
        model_path = os.path.join(base_path, "model.safetensors")
        config_path = os.path.join(base_path, "config.json")
        tokenizer_path = base_path
        
        # Check if required files exist
        if not os.path.exists(model_path):
            st.warning(f"Model file not found: {model_path}")
            return None, None, None
        
        if not os.path.exists(config_path):
            st.warning(f"Config file not found: {config_path}")
            return None, None, None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
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
    
    # Longformer labels: ["sexual", "violence", "substance", "language"]
    # Our format: ["explicit", "violence", "sexual_content", "substance_use"]
    
    # Original default probability scaling - uniform thresholds
    def original_scaling(prob, category):
        """
        Original default scaling approach with uniform 50% threshold for all categories
        Simple linear scaling: probabilities above 50% map linearly to 0-100% scores
        """
        threshold = 0.5  # Default 50% threshold for all categories
        if prob < threshold:
            return 0.0
        else:
            # Linear scaling from threshold to 100%
            return ((prob - threshold) / (1.0 - threshold)) * 100
    
    # Use original default scaling approach
    results = {
        'explicit': original_scaling(probs[3], 'explicit'),      # language -> explicit
        'violence': original_scaling(probs[1], 'violence'),      # violence -> violence  
        'sexual_content': original_scaling(probs[0], 'sexual'),  # sexual -> sexual_content
        'substance_use': original_scaling(probs[2], 'substance'), # substance -> substance_use
        'total_words': len(lyrics.split()),
        'confidence_scores': probs.tolist(),
        'raw_probabilities': probs.tolist(),
        'scaling_method': 'original_default'
    }
    
    # Ensure no negative values and reasonable bounds
    for key in ['explicit', 'violence', 'sexual_content', 'substance_use']:
        results[key] = max(0.0, min(100.0, results[key]))
    
    return results

# Weighted Severity Index
def calculate_content_severity(results):
    weights = {
        'explicit': 0.25,
        'violence': 0.25,
        'sexual_content': 0.25,
        'substance_use': 0.25
    }
    
    severity = sum(results[category] * weights[category] for category in weights)
    return min(100, severity)



# Classification function
def classify_lyrics(lyrics, tokenizer, model, preprocessor):
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
        st.caption("Charts removed for demo version.")

# Main application
def main():
    # Parse arguments for configuration info
    args = parse_arguments()
    
    # Application header
    st.markdown("""
    <div class="flex center-y justify-between" style="margin-bottom:18px;">
        <div>
            <h1 style="margin:0;">LyricLens</h1>
            <div style="font-size:.8rem; color:var(--text-secondary);">AI-Powered Lyric Content Assessment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show configuration info in sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        if args.model_path:
            st.info(f"🎯 Custom model path: `{args.model_path}`")
        else:
            st.info("🎯 Using default model path: `./model`")
        
        st.markdown("### Command Line Usage")
        st.code("""
    # Default usage
    streamlit run app.py

    # Custom model path
    streamlit run app.py -- --model-path /path/to/model

    # Custom model path + port
    streamlit run app.py -- --model-path /path/to/model --port 8502
            """, language="bash")

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
                            lyrics,
                            longformer_tokenizer,
                            longformer_model,
                            longformer_preprocessor
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
    # Check if we're running via streamlit or directly
    import sys
    
    if 'streamlit' in sys.modules or any('streamlit' in arg for arg in sys.argv):
        # Running via Streamlit
        main()
    else:
        # Running directly with Python - show help and start Streamlit
        args = parse_arguments()
        
        print("🎵 LyricLens - AI-Powered Lyric Content Assessment")
        print("=" * 50)
        print(f"Model path: {args.model_path or './model'}")
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print()
        print("Starting Streamlit server...")
        
        # Construct streamlit command
        import subprocess
        cmd = [
            'streamlit', 'run', __file__,
            '--server.port', str(args.port),
            '--server.address', args.host
        ]
        
        if args.model_path:
            cmd.extend(['--', '--model-path', args.model_path])
            
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n👋 LyricLens stopped.")
        except FileNotFoundError:
            print("❌ Streamlit not found. Please install it: pip install streamlit")
            print("Then run: streamlit run app.py")
