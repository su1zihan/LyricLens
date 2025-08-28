# LyricLens: AI-Powered Music Content Analysis System

LyricLens is an open source system for multi label lyric classification and music content rating.  
It evaluates song lyrics in **four categories**: sexual content, violence, explicit language, and substance use.  
The predictions are mapped into a **five level Music Content Rating (MCR)** that provides interpretable and standardized content labels for parents, educators, researchers, and platform developers.  
The system outputs category specific assessments together with overall content ratings that are consistent with widely used content standards.  

## Architecture

The system architecture consists of three main parts as shown below.

1. **Multi label classification**  
   The system predicts four categories of explicit content: sexual content, violence, explicit language, and substance use.  

2. **Music Content Rating (MCR)**  
   The predictions are mapped into five standardized levels with clear thresholds and descriptors:  

   | Rating | Content Description |
   |--------|----------------------|
   | **M-E** (Everyone) | Suitable for all ages; no explicit sexual themes, violence, substance use, or strong language. |
   | **M-P** (Parental Guidance Suggested) | Some material may not be suitable for children; may contain mild language, minimal suggestive themes, or very brief non-graphic violence. |
   | **M-T** (Teen) | Suitable for ages 13+; may contain violence, suggestive themes, drug references, or infrequent use of strong language. |
   | **M-R** (Restricted) | Under 17 requires accompanying adult guardian; may contain intense violence, strong sexual content, frequent strong language, or drug abuse. |
   | **M-AO** (Adults Only) | Suitable only for adults (18+); may include explicit sexual content, extreme violence, or graphic drug abuse. |

3. **Content Severity Index (CSI)**  
   In addition to discrete ratings, the system can output a **numeric score** summarizing the overall severity of explicit content.  
   The CSI is calculated as the average of the four category probabilities (sexual, violence, language, substance).  
   The score ranges from 0 to 100 and offers a quick way to compare songs by their overall explicitness.  
   CSI is an auxiliary score for comparison and does not replace the primary MCR rating.  

## Installation and Setup
### Prerequisites
- **Python 3.10 or higher**
- **8GB+ RAM** (recommended for model loading)
- **Internet connection** (for initial NLTK data downloads)
- **CUDA-compatible GPU** (optional, for accelerated inference)

### Step 1: Clone and Setup

Clone the repository

```bash
git clone https://github.com/su1zihan/LyricLens
```

```bash
cd LyricLens
```

Create virtual environment (optional but recommended)
```bash
python -m venv env
```

macOS/Linux:
```bash
source env/bin/activate
```

Windows (PowerShell or CMD):
```bash
env\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Model Files (Required)
**Important:** The model files are not included in the repository due to their size (3.5GB+).
Download the pre-trained checkpoint from the following link and place it inside a new `model/` directory:

[Download Model](https://drive.google.com/drive/folders/1EQlMFnAieKLeGEQR0ViQdk1Su2P8mjPy?usp=sharing)

Create a `model/` directory and add the following files:
- `model.safetensors` (Longformer model weights)
- `config.json` (Model configuration)
- `tokenizer_config.json` (Tokenizer configuration)
- `vocab.json` (Vocabulary mappings)
- `merges.txt` (BPE merges)
- `special_tokens_map.json` (Special tokens)

### Step 4: Launch Application
```bash
# Default launch (auto-detects model in ./model or ./checkpoint-3588)
streamlit run app.py

# Launch with a custom model path
streamlit run app.py -- --model-path /path/to/your/model

# Launch on a different port
streamlit run app.py -- --port 8502

# Launch with custom model path and port
streamlit run app.py -- --model-path ./model --port 8502
```

The application will be accessible at `http://localhost:8501` by default.
A model checkpoint is required. If --model-path is not provided, the app will search in common folders (./model, ./checkpoint-3588, ./models, ./checkpoint).


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
## Research and Development

This system represents an advancement in automated content analysis for musical media. The combination of transformer-based deep learning with traditional machine learning approaches provides both accuracy and interpretability for content assessment applications.

For research collaborations, technical inquiries, or commercial applications, please contact the project maintainer.
