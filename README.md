# LyricLens: AI-Powered Music Content Analysis System

LyricLens is an open source system for multi label lyric classification and music content rating. It evaluates song lyrics in **four categories**: sexual content, violence, explicit language, and substance use. 

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
   In addition to discrete ratings, the system can output a **numeric score** summarizing the overall severity of explicit content. The CSI is calculated as the average of the four category probabilities (sexual, violence, language, substance). The score ranges from 0 to 100 and offers a quick way to compare songs by their overall explicitness. CSI is an auxiliary score for comparison and does not replace the primary MCR rating.  

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
Download the pre-trained checkpoint from the following link and place it inside a new `model/` directory: [Model Link](https://drive.google.com/drive/folders/1EQlMFnAieKLeGEQR0ViQdk1Su2P8mjPy?usp=sharing)

The download may come as a **zip archive**. After downloading:  
1. Unzip the archive  
2. Create a folder named `model/` inside the project directory (`LyricLens/`)  
3. Copy the all files from the unzipped folder into `LyricLens/model/` 

### Step 4: Launch the application

By default the app will start on http://localhost:8501.  

**Default launch (auto-detects model):**

```bash
streamlit run app.py
```

Launch with a custom model path:
```bash
streamlit run app.py -- --model-path /path/to/your/model
```

Launch on a different port:
```bash
streamlit run app.py -- --port 8502
```

Launch with both custom model path and port:
```bash
streamlit run app.py -- --model-path ./model --port 8502
```

## Usage

### After Launch
1. Open http://localhost:8501 in your browser  
2. Paste song lyrics into the input area  
3. Click **Analyze**  
4. View the results: category scores, CSI score, and final MCR rating  

### Command-Line Help
You can check all available options with:

```bash
streamlit run app.py -- --help
```

## Applications

### Primary Use Cases
- **Parental Controls**: Automated content screening for age-appropriate music selection
- **Educational Settings**: Content assessment for classroom and institutional use
- **Content Creation**: Compliance verification for streaming platforms and media distribution
- **Research Applications**: Large-scale content analysis for academic and industry research

## Research and Development

This system represents an advancement in automated content analysis for musical media.  
The combination of transformer-based deep learning with traditional machine learning approaches provides both accuracy and interpretability for content assessment applications.  

For research collaborations, technical inquiries, or commercial applications, please contact the project maintainer.  

## Authors

- **Kai-Yu Lu** — Khoury College of Computer Sciences, Northeastern University, Seattle, WA  
  ✉️ [lu.kaiy@northeastern.edu](mailto:lu.kaiy@northeastern.edu)

- **Zihan Su** — Khoury College of Computer Sciences, Northeastern University, Seattle, WA  
  ✉️ [su.zihan1@northeastern.edu](mailto:su.zihan1@northeastern.edu)


- **Malhar Sham Ghogare** — Khoury College of Computer Sciences, Northeastern University, Seattle, WA  
  ✉️ [ghogare.m@northeastern.edu](mailto:ghogare.m@northeastern.edu)


- **Shanu Sushmita** — Khoury College of Computer Sciences, Northeastern University, Seattle, WA  
  ✉️ [s.sushmita.m@northeastern.edu](mailto:s.sushmita.m@northeastern.edu)

