# LyricLens: AI-Powered Music Content Analysis System

LyricLens is a comprehensive content analysis platform that leverages advanced machine learning to evaluate song lyrics for safety and appropriateness. The system provides automated content classification, severity assessment, and standardized music content ratings for parents, educators, content creators, and researchers.

## Overview

The platform combines state-of-the-art transformer models with traditional machine learning approaches to deliver multi-dimensional content analysis. LyricLens processes song lyrics through sophisticated natural language processing pipelines and outputs both granular category-specific assessments and standardized content ratings compatible with industry frameworks.

## System Architecture

### Core Components

**Longformer Model Pipeline**
- Fine-tuned transformer architecture optimized for long-form text analysis
- Multi-label classification across four content categories: violence, explicit language, sexual content, and substance use
- Supports input sequences up to 4096 tokens for comprehensive lyric analysis
- Model checkpoint includes weights, configuration, and tokenizer files for deployment

**Content Severity Index**
The Content Severity Index synthesizes multi-categorical assessment into a single metric using equal-weight aggregation: S = Σ(0.25 × pi) across violence, explicit language, sexual content, and substance use categories. This provides rapid dimensional reduction with balanced assessment, offering intuitive 0-100 interpretability and real-time computational efficiency. The equal weighting prevents category dominance while supporting diverse applications from parental filtering to music analytics research.

**Music Content Rating System**
- Standardized rating framework with five classifications: M-E (Everyone), M-P (Parental Guidance), M-T (Teen), M-R (Restricted), M-AO (Adults Only)
- Threshold-based rating assignment with special conditions for extreme content
- Content descriptors for granular understanding of rating rationale


| Rating | Content Description |
|--------|----------------------|
| **M-E** (Everyone) | Suitable for all ages; no explicit sexual themes, violence, substance use, or strong language. |
| **M-P** (Parental Guidance Suggested) | Some material may not be suitable for children; may contain mild language, minimal suggestive themes, or very brief non-graphic violence. |
| **M-T** (Teen) | Suitable for ages 13+; may contain violence, suggestive themes, drug references, or infrequent use of strong language. |
| **M-R** (Restricted) | Under 17 requires accompanying adult guardian; may contain intense violence, strong sexual content, frequent strong language, or drug abuse. |
| **M-AO** (Adults Only) | Suitable only for adults (18+); may include explicit sexual content, extreme violence, or graphic drug abuse. |


**Text Processing Pipeline**
- Advanced preprocessing including contraction expansion, lemmatization, and normalization
- Multi-stage tokenization optimized for lyrical content
- Robust handling of incomplete or ambiguous input through fallback mechanisms

### Technical Implementation

**User Interface**
- Streamlit-based web application with responsive design
- Real-time analysis with progress indicators and detailed result visualization
- Compact, conference-ready interface with professional styling

**Model Integration**
- Primary analysis via fine-tuned Longformer model with safetensors format
- Category-specific probability scaling with configurable thresholds
- Caching mechanisms for optimized model loading and inference performance

**Supporting Models**
- TF-IDF vectorization with XGBoost classification for auxiliary analysis
- Feature importance analysis for interpretability and model validation
- Fallback classification pipeline for system resilience

## Installation and Setup

### Prerequisites
- **Python 3.10 or higher**
- **8GB+ RAM** (recommended for model loading)
- **Internet connection** (for initial NLTK data downloads)
- **CUDA-compatible GPU** (optional, for accelerated inference)

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd kids-safe-music-INTEGRATED

# Create virtual environment
python -m venv lyrics_analyzer_env

# Activate virtual environment
# Windows:
lyrics_analyzer_env\Scripts\activate
# macOS/Linux:
source lyrics_analyzer_env/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Model Files (Required)
**Important:** The model files are not included in the repository due to their size (3.5GB+).

Create a `model/` directory and add the following files:
- `model.safetensors` (Longformer model weights)
- `config.json` (Model configuration)
- `tokenizer_config.json` (Tokenizer configuration)
- `vocab.json` (Vocabulary mappings)
- `merges.txt` (BPE merges)
- `special_tokens_map.json` (Special tokens)

### Step 4: Launch Application
```bash
# Default launch (uses ./checkpoint-3588)
streamlit run app.py

# Custom model path
streamlit run app.py -- --model-path /path/to/your/model

# Custom port
streamlit run app.py -- --port 8502
```

The application will be accessible at `http://localhost:8501`

## Project Structure

```
├── app.py                           # Main application with UI and analysis logic
├── music_content_rating.py          # Content rating system implementation
├── train_model.py                   # Model training and validation scripts
├── requirements.txt                 # Python dependencies specification
├── violence_top_features.txt        # Feature importance analysis results
├── enhanced_tfidf_vectorizer.pkl    # Pre-trained TF-IDF vectorizer
├── enhanced_xgboost_model.pkl       # Pre-trained XGBoost classifier
├── checkpoint-3588/                 # Longformer model checkpoint
│   ├── config.json                  # Model configuration
│   ├── model.safetensors           # Model weights
│   ├── tokenizer_config.json       # Tokenizer configuration
│   └── vocab.json                   # Vocabulary mappings
└── lyrics_analyzer_env/             # Python virtual environment
```

## Usage

### Command-Line Options
```bash
# Show help
python app.py --help

# Custom model directory
streamlit run app.py -- --model-path ./my-model-checkpoint

# Production deployment
streamlit run app.py -- --host 0.0.0.0 --port 8080 --model-path /opt/models/lyriclens
```

### Basic Analysis
1. Launch the application and navigate to the web interface
2. Input song lyrics in the provided text area
3. Click "Analyze" to process the content
4. Review results including category scores, severity index, and content rating

### Advanced Features
- Real-time content scoring across multiple dimensions
- Standardized music content ratings with detailed explanations
- Visual severity indicators and category breakdowns
- Export capabilities for research and documentation

## Model Details

### Longformer Architecture
- 12-layer transformer with 768 hidden dimensions
- 12 attention heads with combined local and global attention mechanisms
- Pre-trained on large text corpora and fine-tuned on lyric-specific datasets
- Optimized for multi-label classification with four target categories

### Training Framework
- PyTorch implementation with Hugging Face Transformers
- Safetensors format for secure and efficient model serialization
- Comprehensive training state preservation for reproducibility

### Performance Characteristics
- Real-time inference capability for production deployment
- Scalable architecture supporting batch processing
- Robust handling of variable-length input sequences

---

The pre-trained model can be downloaded from the following link: [Download Model](https://drive.google.com/drive/folders/1EQlMFnAieKLeGEQR0ViQdk1Su2P8mjPy?usp=sharing)

## Applications

### Primary Use Cases
- **Parental Controls**: Automated content screening for age-appropriate music selection
- **Educational Settings**: Content assessment for classroom and institutional use
- **Content Creation**: Compliance verification for streaming platforms and media distribution
- **Research Applications**: Large-scale content analysis for academic and industry research

### Integration Capabilities
- RESTful API potential for third-party system integration
- Batch processing capabilities for large-scale analysis
- Configurable thresholds for diverse organizational requirements

## Technical Requirements

### Dependencies & Requirements
```txt
streamlit>=1.27.0          # Web application framework
torch>=1.9.0               # Deep learning backend
transformers>=4.21.0       # Hugging Face model library
safetensors>=0.3.0         # Secure model serialization
nltk>=3.8.1               # Natural language processing
pandas>=2.1.0             # Data manipulation
numpy>=1.24.3             # Numerical computing
scikit-learn>=1.3.0       # Machine learning utilities
```

### System Requirements
- Minimum 8GB RAM for optimal performance
- GPU acceleration recommended for large-scale deployments
- Network connectivity for initial model downloads

## Research and Development

This system represents an advancement in automated content analysis for musical media. The combination of transformer-based deep learning with traditional machine learning approaches provides both accuracy and interpretability for content assessment applications.

For research collaborations, technical inquiries, or commercial applications, please contact the project maintainer.
