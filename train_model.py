import pandas as pd
import numpy as np
import re
import nltk
import pickle
import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.utils import resample
import matplotlib.pyplot as plt
import ast
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define this class at the module level, not inside a function
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

# Enhanced preprocessing function
def preprocess_text_enhanced(text):
    if pd.isnull(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle contractions more systematically
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        "'m": " am", "it's": "it is", "that's": "that is"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove special characters but keep some meaningful punctuation temporarily
    text = re.sub(r'[^\w\s\!\?\.]', ' ', text)
    
    # Handle repeated characters (like "sooooo" -> "so")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords but keep some important ones for context
    stop_words = set(stopwords.words('english'))
    # Keep some stopwords that might be important for content classification
    keep_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
    stop_words -= keep_words
    
    filtered_tokens = [word for word in tokens if word.isalpha() and (word not in stop_words or len(word) > 2)]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)
    
    return preprocessed_text

def parse_extracted_windows(windows_str):
    """Parse the string representation of windows list."""
    if pd.isnull(windows_str) or windows_str == '':
        return []
    
    try:
        # Replace single quotes with double quotes for valid JSON
        windows_str = windows_str.replace("'", '"')
        return json.loads(windows_str)
    except:
        try:
            # Try ast.literal_eval as a fallback
            return ast.literal_eval(windows_str)
        except:
            print(f"Error parsing: {windows_str[:50]}...")
            return []

# Function to plot class distribution
def plot_class_distribution(df, title='Class Distribution'):
    counts = [
        df['explicit'].sum(),
        df['violence'].sum(),
        df['sex'].sum(), 
        df['substance_use'].sum()
    ]
    
    labels = ['Explicit', 'Violence', 'Sex', 'Substance Use']
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['#1DB954', '#E81123', '#FF8C00', '#4299E1'])
    plt.title(title)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def validate_vectorizer(vectorizer, sample_text="test text"):
    """Validate that vectorizer is properly fitted"""
    try:
        # Check if vectorizer has required attributes
        if not hasattr(vectorizer, 'idf_'):
            print("❌ Vectorizer missing 'idf_' attribute - not fitted!")
            return False
        
        if not hasattr(vectorizer, 'vocabulary_'):
            print("❌ Vectorizer missing 'vocabulary_' attribute - not fitted!")
            return False
        
        if vectorizer.idf_ is None:
            print("❌ Vectorizer 'idf_' is None - not fitted!")
            return False
        
        if not vectorizer.vocabulary_:
            print("❌ Vectorizer vocabulary is empty - not fitted!")
            return False
        
        # Test transform
        test_transform = vectorizer.transform([sample_text])
        print(f"✅ Vectorizer validation passed!")
        print(f"   - Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"   - IDF shape: {vectorizer.idf_.shape}")
        print(f"   - Test transform shape: {test_transform.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vectorizer validation failed: {str(e)}")
        return False

def balance_dataset(X, y, label_idx, method='undersample'):
    """Balance dataset for a specific label"""
    # Get positive and negative samples
    pos_mask = y[:, label_idx] == 1
    neg_mask = y[:, label_idx] == 0
    
    pos_count = np.sum(pos_mask)
    neg_count = np.sum(neg_mask)
    
    print(f"   Original - Positive: {pos_count}, Negative: {neg_count}")
    
    if method == 'undersample' and neg_count > pos_count * 3:
        # Undersample majority class if very imbalanced
        target_neg_count = min(pos_count * 3, neg_count)
        
        neg_indices = np.where(neg_mask)[0]
        selected_neg_indices = np.random.choice(neg_indices, target_neg_count, replace=False)
        pos_indices = np.where(pos_mask)[0]
        
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])
        np.random.shuffle(selected_indices)
        
        print(f"   Balanced - Positive: {pos_count}, Negative: {target_neg_count}")
        return selected_indices
    
    return np.arange(len(X))

def optimize_xgboost_params(X_train, y_train, label_name):
    """Find optimal XGBoost parameters using GridSearch"""
    print(f"   Optimizing parameters for {label_name}...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Use smaller grid for faster training
    if X_train.shape[0] > 20000:  # Large dataset
        param_grid = {
            'n_estimators': [150, 250],
            'max_depth': [4, 6],
            'learning_rate': [0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    
    # Calculate class weights
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Base estimator
    base_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
    
    # Use StratifiedKFold for better cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # GridSearch with F1 score as metric (better for imbalanced data)
    grid_search = GridSearchCV(
        base_estimator, 
        param_grid, 
        cv=cv, 
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    print("🚀 Starting Enhanced Model Training...")
    
    # Check if dataset exists
    dataset_path = 'combined_dataset_with_windows.csv'
    if not pd.io.common.file_exists(dataset_path):
        print(f"❌ Dataset file '{dataset_path}' not found!")
        return
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} rows")
    
    # Check required columns
    required_columns = ['extracted_windows', 'explicit', 'violence', 'sex', 'substance_use']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return
    
    # Convert explicit column if needed
    if df['explicit'].dtype == 'object':
        df['explicit'] = df['explicit'].map({'True': True, 'False': False})
    
    # Show class distribution
    print("\n📊 Original Class Distribution:")
    for col in ['explicit', 'violence', 'sex', 'substance_use']:
        count = df[col].astype(int).sum()
        percentage = (count / len(df)) * 100
        print(f"{col.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Process windows with enhanced preprocessing
    print("\n🔄 Processing extracted windows with enhanced preprocessing...")
    windows = []
    window_labels = []
    processed_rows = 0
    
    for i, row in df.iterrows():
        row_windows = parse_extracted_windows(row['extracted_windows'])
        
        if not row_windows:
            continue
            
        for window in row_windows:
            if window and len(window.strip()) > 0:
                # Apply enhanced preprocessing
                enhanced_window = preprocess_text_enhanced(window)
                if enhanced_window and len(enhanced_window.split()) >= 3:  # Min 3 words
                    windows.append(enhanced_window)
                    window_labels.append([
                        row['explicit'] if isinstance(row['explicit'], bool) else row['explicit'] == 'True',
                        row['violence'] == 1,
                        row['sex'] == 1, 
                        row['substance_use'] == 1
                    ])
        
        processed_rows += 1
        if processed_rows % 1000 == 0:
            print(f"   Processed {processed_rows} rows...")
    
    window_labels = np.array(window_labels).astype(int)
    print(f"✅ Extracted {len(windows)} enhanced windows for training")
    
    if len(windows) == 0:
        print("❌ No windows extracted!")
        return
    
    # Show window-level class distribution
    print("\n📊 Window-level Class Distribution:")
    label_names = ['Explicit', 'Violence', 'Sex', 'Substance Use']
    for i, name in enumerate(label_names):
        count = np.sum(window_labels[:, i])
        percentage = (count / len(window_labels)) * 100
        print(f"{name}: {count} ({percentage:.1f}%)")
    
    # Split data
    print("\n🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        windows, window_labels, test_size=0.2, random_state=42, 
        stratify=window_labels[:, 1]  # Stratify by violence (most balanced)
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create enhanced TF-IDF vectorizer
    print("\n🔤 Creating enhanced TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=8000,  # Increased from 5000
        min_df=3,           # Reduced from 5 for more vocabulary
        max_df=0.8,         # Increased from 0.7
        ngram_range=(1, 3), # Added trigrams
        sublinear_tf=True,  # Better for large feature spaces
        use_idf=True,
        smooth_idf=True,
        norm='l2'
    )
    
    print("Fitting vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Validate vectorizer
    if not validate_vectorizer(vectorizer, X_train[0] if X_train else "test"):
        print("❌ Vectorizer validation failed!")
        return
    
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF features shape: {X_train_tfidf.shape}")
    
    # Train optimized models for each label
    print("\n🎯 Training optimized models...")
    models = []
    
    for i, label_name in enumerate(label_names):
        print(f"\n🔥 Training {label_name} classifier...")
        
        # Get labels for this classification task
        y_label_train = y_train[:, i]
        y_label_test = y_test[:, i]
        
        # Balance dataset if very imbalanced
        pos_count = np.sum(y_label_train)
        neg_count = len(y_label_train) - pos_count
        
        if neg_count > pos_count * 4:  # Very imbalanced
            print(f"   Dataset is very imbalanced, applying balancing...")
            selected_indices = balance_dataset(X_train, y_train, i, method='undersample')
            X_balanced = X_train_tfidf[selected_indices]
            y_balanced = y_label_train[selected_indices]
        else:
            X_balanced = X_train_tfidf
            y_balanced = y_label_train
        
        # Optimize parameters
        optimized_model = optimize_xgboost_params(X_balanced, y_balanced, label_name)
        
        # Special adjustment for violence (make it less sensitive)
        if i == 1:  # Violence
            optimized_model.scale_pos_weight *= 0.7
            print(f"   Applied violence sensitivity adjustment")
        
        models.append(optimized_model)
        
        # Evaluate on test set
        y_pred = optimized_model.predict(X_test_tfidf)
        y_pred_proba = optimized_model.predict_proba(X_test_tfidf)[:, 1]
        
        accuracy = accuracy_score(y_label_test, y_pred)
        f1 = f1_score(y_label_test, y_pred)
        
        try:
            auc = roc_auc_score(y_label_test, y_pred_proba)
            print(f"✅ {label_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        except:
            print(f"✅ {label_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        print(f"\nDetailed classification report for {label_name}:")
        print(classification_report(y_label_test, y_pred))
    
    # Create multi-label model
    multi_model = MultiLabelXGBoostClassifier(models)
    
    # Save models
    print("\n💾 Saving enhanced models...")
    
    # Final validation
    if not validate_vectorizer(vectorizer):
        print("❌ Final vectorizer validation failed!")
        return
    
    try:
        joblib.dump(vectorizer, 'enhanced_tfidf_vectorizer.pkl')
        joblib.dump(multi_model, 'enhanced_xgboost_model.pkl')
        
        print("✅ Enhanced models saved successfully!")
        
        # Test loading
        loaded_vectorizer = joblib.load('enhanced_tfidf_vectorizer.pkl')
        loaded_model = joblib.load('enhanced_xgboost_model.pkl')
        
        if validate_vectorizer(loaded_vectorizer):
            print("✅ Saved models validated successfully!")
        
        # Test prediction
        sample_text = "This is a test lyric with some explicit language damn"
        sample_tfidf = loaded_vectorizer.transform([sample_text])
        sample_prediction = loaded_model.predict(sample_tfidf)
        
        print(f"✅ Test prediction: {sample_prediction}")
        
    except Exception as e:
        print(f"❌ Error saving models: {str(e)}")
        return
    
    print("\n🎉 Enhanced training complete!")
    print("\nEnhanced model files created:")
    print("- enhanced_tfidf_vectorizer.pkl")
    print("- enhanced_xgboost_model.pkl")
    
    print("\n💡 Improvements made:")
    print("✅ Enhanced text preprocessing")
    print("✅ Larger vocabulary (8000 features)")
    print("✅ Added trigrams for better context")
    print("✅ Hyperparameter optimization")
    print("✅ Dataset balancing for very imbalanced classes")
    print("✅ Better evaluation metrics (F1, AUC)")

if __name__ == "__main__":
    main()