"""
Universal NSL-KDD Preprocessing Module
=====================================

Enhanced NSL-KDD preprocessing that works with ANY model:
- Mateen (AutoEncoder ensemble) 
- CopulaGAN (Synthetic data generation)
- Traditional ML (XGBoost, RF, SVM, etc.)
- Deep Learning (PyTorch, TensorFlow)
- Custom models

Features:
- Consensus feature selection from multiple algorithms
- AutoEncoder-optimized preprocessing  
- CopulaGAN-optimized preprocessing
- Multiple output formats (numpy, pandas, PyTorch)
- Flexible sampling strategies
- Robust outlier handling and scaling

Author: Universal preprocessing for all ML frameworks
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========================================
# UNIFIED PREPROCESSING CLASS
# ========================================

class UniversalNSLKDDPreprocessor:
    """
    Universal NSL-KDD preprocessor that works with any ML framework
    """
    
    def __init__(self, n_features=15, sampling_strategy='smote_tomek', scaler_type='robust'):
        self.n_features = n_features
        self.sampling_strategy = sampling_strategy
        self.scaler_type = scaler_type
        self.label_encoders = {}
        self.scaler = None
        self.selected_features = None
        
    def load_and_clean_data(self, file_path_or_train_path, test_path=None):
        """
        Load and perform initial cleaning of NSL-KDD data
        Supports both single file (CopulaGAN) and train/test files (Mateen)
        """
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
        ]
        
        if test_path is None:
            # Single file mode (CopulaGAN)
            df = pd.read_csv(file_path_or_train_path, names=columns)
            
            # Drop unnecessary columns
            drop_cols = ['difficulty_level', 'num_outbound_cmds', 'is_host_login']
            df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
            
            # Convert labels to binary
            df['label'] = df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
            
            return df
        else:
            # Train/test mode (Mateen)
            train_df = pd.read_csv(file_path_or_train_path, names=columns)
            test_df = pd.read_csv(test_path, names=columns)
            
            # Drop unnecessary columns
            drop_cols = ['difficulty_level', 'num_outbound_cmds', 'is_host_login']
            train_df = train_df.drop([col for col in drop_cols if col in train_df.columns], axis=1)
            test_df = test_df.drop([col for col in drop_cols if col in test_df.columns], axis=1)
            
            # Convert labels to binary
            train_df['label'] = train_df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
            test_df['label'] = test_df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
            
            return train_df, test_df
    
    def create_engineered_features(self, df):
        """Create new engineered features optimized for anomaly detection"""
        df_eng = df.copy()
        
        # Network flow features
        df_eng['bytes_ratio'] = np.where(df_eng['dst_bytes'] > 0, 
                                       df_eng['src_bytes'] / df_eng['dst_bytes'], 0)
        df_eng['total_bytes'] = df_eng['src_bytes'] + df_eng['dst_bytes']
        df_eng['bytes_per_second'] = np.where(df_eng['duration'] > 0,
                                            df_eng['total_bytes'] / df_eng['duration'], 0)
        
        # Connection pattern features
        df_eng['srv_ratio'] = np.where(df_eng['count'] > 0,
                                     df_eng['srv_count'] / df_eng['count'], 0)
        df_eng['error_ratio'] = df_eng['serror_rate'] + df_eng['rerror_rate']
        df_eng['srv_error_ratio'] = df_eng['srv_serror_rate'] + df_eng['srv_rerror_rate']
        
        # Host-based features
        df_eng['host_srv_ratio'] = np.where(df_eng['dst_host_count'] > 0,
                                          df_eng['dst_host_srv_count'] / df_eng['dst_host_count'], 0)
        df_eng['host_error_total'] = (df_eng['dst_host_serror_rate'] + 
                                    df_eng['dst_host_rerror_rate'])
        
        # Security-related features
        df_eng['security_score'] = (df_eng['num_failed_logins'] + 
                                  df_eng['num_compromised'] + 
                                  df_eng['num_root'])
        df_eng['file_activity'] = df_eng['num_file_creations'] + df_eng['num_shells']
        
        # Log transformations for skewed features
        log_features = ['src_bytes', 'dst_bytes', 'count', 'srv_count', 'dst_host_count']
        for feature in log_features:
            if feature in df_eng.columns:
                df_eng[f'{feature}_log'] = np.log1p(df_eng[feature])
        
        return df_eng
    
    def handle_outliers(self, X_train, X_test=None, method='iqr'):
        """Handle outliers using IQR method"""
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy() if X_test is not None else None
        
        numeric_cols = X_train_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = X_train_clean[col].quantile(0.25)
                Q3 = X_train_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                X_train_clean[col] = X_train_clean[col].clip(lower_bound, upper_bound)
                if X_test_clean is not None:
                    X_test_clean[col] = X_test_clean[col].clip(lower_bound, upper_bound)
        
        return X_train_clean, X_test_clean
    
    def encode_categorical_features(self, X_train, X_test=None):
        """Enhanced categorical encoding with frequency features"""
        categorical_columns = ['protocol_type', 'service', 'flag']
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy() if X_test is not None else None
        
        for col in categorical_columns:
            if col not in X_train.columns:
                continue
                
            # Standard label encoding
            if col not in self.label_encoders:
                le = LabelEncoder()
                X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                X_train_encoded[col] = le.transform(X_train[col].astype(str))
            
            if X_test is not None:
                # Handle unseen categories in test set
                test_values = X_test[col].astype(str)
                for val in test_values.unique():
                    if val not in le.classes_:
                        test_values = test_values.replace(val, le.classes_[0])
                X_test_encoded[col] = le.transform(test_values)
            
            # Add frequency encoding
            freq_map = X_train[col].value_counts().to_dict()
            X_train_encoded[f'{col}_freq'] = X_train[col].map(freq_map)
            if X_test is not None:
                X_test_encoded[f'{col}_freq'] = X_test[col].map(freq_map).fillna(0)
        
        return X_train_encoded, X_test_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using specified scaler"""
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        
        if self.scaler is None:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            
            # Fit on training data
            self.scaler.fit(X_train[numeric_columns])
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy() if X_test is not None else None
        
        X_train_scaled[numeric_columns] = self.scaler.transform(X_train[numeric_columns])
        if X_test is not None:
            X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        
        return X_train_scaled, X_test_scaled
    
    def consensus_feature_selection(self, X_train, y_train, X_test=None):
        """Advanced consensus feature selection from multiple algorithms"""
        
        # 1. XGBoost feature importance
        xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        xgb_scores = xgb.feature_importances_
        
        # 2. Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_scores = rf.feature_importances_
        
        # 3. Mutual Information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        # 4. F-score (ANOVA)
        f_scores = f_classif(X_train, y_train)[0]
        
        # Normalize all scores
        xgb_norm = xgb_scores / np.max(xgb_scores)
        rf_norm = rf_scores / np.max(rf_scores)
        mi_norm = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
        f_norm = f_scores / np.max(f_scores)
        
        # Weighted combination (equal weights for consensus)
        combined_scores = 0.25 * xgb_norm + 0.25 * rf_norm + 0.25 * mi_norm + 0.25 * f_norm
        
        # Select top features
        feature_indices = np.argsort(combined_scores)[-self.n_features:]
        self.selected_features = X_train.columns[feature_indices].tolist()
        
        print(f"Selected {len(self.selected_features)} features using consensus:")
        for i, feature in enumerate(self.selected_features, 1):
            print(f"  {i:2d}. {feature}")
        
        X_train_selected = X_train.iloc[:, feature_indices]
        X_test_selected = X_test.iloc[:, feature_indices] if X_test is not None else None
        
        return X_train_selected, X_test_selected
    
    def apply_sampling(self, X_train, y_train):
        """Apply sampling strategy if needed"""
        if self.sampling_strategy == 'none':
            return X_train, y_train
        elif self.sampling_strategy == 'smote':
            sampler = SMOTE(random_state=42)
        elif self.sampling_strategy == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif self.sampling_strategy == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=42)
        elif self.sampling_strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        else:
            return X_train, y_train
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def fit_transform(self, file_path_or_train_path, test_path=None):
        """
        Complete preprocessing pipeline
        Supports both single file mode and train/test mode
        """
        print("=== Universal NSL-KDD Preprocessing ===")
        
        # Step 1: Load and clean
        print("1. Loading and cleaning data...")
        if test_path is None:
            # Single file mode (CopulaGAN)
            df = self.load_and_clean_data(file_path_or_train_path)
            X_train = df.drop('label', axis=1)
            y_train = df['label']
            X_test = None
            y_test = None
            print(f"   Loaded: {df.shape}")
        else:
            # Train/test mode (Mateen)
            train_df, test_df = self.load_and_clean_data(file_path_or_train_path, test_path)
            X_train = train_df.drop('label', axis=1)
            y_train = train_df['label']
            X_test = test_df.drop('label', axis=1)
            y_test = test_df['label']
            print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Step 2: Feature engineering
        print("2. Engineering features...")
        X_train = self.create_engineered_features(X_train)
        if X_test is not None:
            X_test = self.create_engineered_features(X_test)
        print(f"   After engineering: {X_train.shape}")
        
        # Step 3: Handle outliers
        print("3. Handling outliers...")
        X_train, X_test = self.handle_outliers(X_train, X_test)
        
        # Step 4: Encode categorical
        print("4. Encoding categorical features...")
        X_train, X_test = self.encode_categorical_features(X_train, X_test)
        
        # Step 5: Scale features
        print("5. Scaling features...")
        X_train, X_test = self.scale_features(X_train, X_test)
        
        # Step 6: Feature selection
        print(f"6. Selecting top {self.n_features} features...")
        X_train, X_test = self.consensus_feature_selection(X_train, y_train, X_test)
        
        # Step 7: Sampling (optional)
        if self.sampling_strategy != 'none':
            print(f"7. Applying {self.sampling_strategy} sampling...")
            X_train, y_train = self.apply_sampling(X_train, y_train)
        
        print(f"   Final: Train {X_train.shape}")
        if X_test is not None:
            print(f"          Test {X_test.shape}")
        print(f"   Class balance: Normal {np.sum(y_train==0)}, Attack {np.sum(y_train==1)}")
        
        if test_path is None:
            # Single file mode - return combined dataframe
            df_final = X_train.copy()
            df_final['label'] = y_train.values
            return df_final
        else:
            # Train/test mode - return separate arrays
            return X_train, y_train, X_test, y_test


# ========================================
# UNIVERSAL INTERFACE FUNCTIONS
# ========================================

def preprocess_for_any_model(train_path, test_path=None, model_type='general', n_features=15, **kwargs):
    """
    Universal preprocessing function that adapts to any model type
    
    Args:
        train_path: Path to training data (or single file for CopulaGAN)
        test_path: Path to test data (None for CopulaGAN single file mode)
        model_type: 'autoencoder', 'copulagan', 'traditional_ml', 'deep_learning', 'general'
        n_features: Number of features to select
        **kwargs: Additional arguments for preprocessor
    
    Returns:
        Preprocessed data in appropriate format for the model type
    """
    
    # Configure preprocessing based on model type
    if model_type == 'autoencoder':
        # Optimized for autoencoders (like Mateen)
        sampling = 'none'  # Don't balance for AE training
        scaler = 'robust'  # Better for outliers
    elif model_type == 'copulagan':
        # Optimized for CopulaGAN
        sampling = 'none'  # CopulaGAN handles imbalance internally
        scaler = 'robust'  # Best for cybersecurity data
    elif model_type == 'traditional_ml':
        # Optimized for traditional ML
        sampling = 'smote_tomek'  # Balance classes
        scaler = 'standard'  # Standard scaling
    elif model_type == 'deep_learning':
        # Optimized for deep learning
        sampling = 'smote'  # Light balancing
        scaler = 'minmax'  # Bounded features
    else:  # general
        sampling = 'smote_tomek'
        scaler = 'robust'
    
    # Override with user preferences
    sampling = kwargs.get('sampling_strategy', sampling)
    scaler = kwargs.get('scaler_type', scaler)
    
    # Create preprocessor
    preprocessor = UniversalNSLKDDPreprocessor(
        n_features=n_features,
        sampling_strategy=sampling,
        scaler_type=scaler
    )
    
    # Process data
    if test_path is None:
        # Single file mode (CopulaGAN)
        df = preprocessor.fit_transform(train_path)
        return df, preprocessor
    else:
        # Train/test mode
        X_train, y_train, X_test, y_test = preprocessor.fit_transform(train_path, test_path)
        
        # Return in appropriate format
        if model_type == 'autoencoder':
            # Return only normal samples for training + full test set
            normal_mask = y_train == 0
            X_train_normal = X_train[normal_mask]
            return X_train_normal, X_test, y_test, preprocessor.selected_features, preprocessor
        else:
            # Return full training set
            return X_train, y_train, X_test, y_test, preprocessor.selected_features, preprocessor


# ========================================
# COPULAGAN COMPATIBILITY
# ========================================

def prepare_data_for_copulagan(train_path, n_features=15):
    """
    Prepare data specifically for CopulaGAN using unified preprocessing
    
    Parameters:
    -----------
    train_path : str
        Path to NSL-KDD training data
    n_features : int
        Number of features to select
    
    Returns:
    --------
    df : pandas.DataFrame
        Preprocessed dataframe ready for CopulaGAN
    preprocessor : UniversalNSLKDDPreprocessor
        The fitted preprocessor object
    """
    
    print("🔧 UNIFIED NSL-KDD PREPROCESSING FOR COPULAGAN")
    print("=" * 50)
    
    return preprocess_for_any_model(train_path, test_path=None, model_type='copulagan', n_features=n_features)


# ========================================
# MATEEN COMPATIBILITY FUNCTIONS  
# ========================================

def prepare_datasets(x_train, y_train):
    """For initial training in main.py"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    if hasattr(x_train, 'values'):
        x_train = x_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    X_tensor = torch.FloatTensor(x_train)
    dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder: input = output
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    benign_train = x_train[y_train == 0] if y_train is not None else x_train
    return train_loader, benign_train


def loading_datasets(x_data):
    """For continued training in main.py"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    if hasattr(x_data, 'values'):
        x_data = x_data.values
    
    X_tensor = torch.FloatTensor(x_data)
    dataset = TensorDataset(X_tensor, X_tensor)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    return data_loader, None


def prepare_data(*args, **kwargs):
    """Smart prepare_data for different call patterns"""
    
    # String argument - dataset loading
    if len(args) == 1 and isinstance(args[0], str):
        scenario = args[0]
        if scenario in ["NSLKDD", "NSL-KDD"]:
            # Update these paths to match your file locations
            train_path = "/content/drive/MyDrive/Colab_Projects/Mateen/KDDTrain+.txt"
            test_path = "/content/drive/MyDrive/Colab_Projects/Mateen/KDDTest+.txt"
            
            X_train_normal, X_test, y_test, features, _ = preprocess_for_any_model(
                train_path, test_path, model_type='autoencoder', n_features=15
            )
            
            x_train = X_train_normal.values
            x_test = X_test.values
            y_train = np.zeros(len(x_train))
            y_test = y_test.values
            
            print(f"Enhanced {scenario}: Train {x_train.shape}, Test {x_test.shape}")
            return x_train, x_test, y_train, y_test
        else:
            raise NotImplementedError(f"Dataset {scenario} not implemented")
    
    # Array argument - data loading
    elif len(args) == 1 and hasattr(args[0], 'shape'):
        return loading_datasets(args[0])
    
    # Two arguments - dataset preparation  
    elif len(args) == 2:
        return prepare_datasets(args[0], args[1])
    
    else:
        raise ValueError(f"Unsupported prepare_data call")


def partition_array(x_data, y_data, slice_size):
    """Partition data into slices for streaming analysis"""
    n_samples = len(x_data)
    n_slices = n_samples // slice_size
    
    x_slices, y_slices = [], []
    
    for i in range(n_slices):
        start_idx = i * slice_size
        end_idx = start_idx + slice_size
        x_slices.append(x_data[start_idx:end_idx])
        y_slices.append(y_data[start_idx:end_idx])
    
    if n_samples % slice_size != 0:
        x_slices.append(x_data[n_slices * slice_size:])
        y_slices.append(y_data[n_slices * slice_size:])
    
    print(f"Data partitioned into {len(x_slices)} slices")
    return x_slices, y_slices


# ========================================
# SKLEARN COMPATIBILITY
# ========================================

def get_data_for_sklearn(train_path, test_path=None, n_features=15, balance=True):
    """Get data ready for sklearn models"""
    sampling = 'smote_tomek' if balance else 'none'
    
    if test_path is None:
        # Single file mode - split internally
        df, _ = preprocess_for_any_model(
            train_path, test_path=None, model_type='traditional_ml', 
            n_features=n_features, sampling_strategy=sampling
        )
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train.values, y_train.values, X_test.values, y_test.values
    else:
        # Train/test files provided
        X_train, y_train, X_test, y_test, features, _ = preprocess_for_any_model(
            train_path, test_path, model_type='traditional_ml', 
            n_features=n_features, sampling_strategy=sampling
        )
        
        return X_train.values, y_train.values, X_test.values, y_test.values


# ========================================
# PYTORCH COMPATIBILITY  
# ========================================

def get_data_for_pytorch(train_path, test_path=None, n_features=15, batch_size=1024, balance=True):
    """Get PyTorch DataLoaders for deep learning"""
    sampling = 'smote' if balance else 'none'
    
    if test_path is None:
        # Single file mode
        df, _ = preprocess_for_any_model(
            train_path, test_path=None, model_type='deep_learning',
            n_features=n_features, sampling_strategy=sampling
        )
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Split and convert to tensors
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.LongTensor(y_test.values)
    else:
        # Train/test files provided
        X_train, y_train, X_test, y_test, features, _ = preprocess_for_any_model(
            train_path, test_path, model_type='deep_learning',
            n_features=n_features, sampling_strategy=sampling
        )
        
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.LongTensor(y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train_tensor.shape[1]


def get_data_for_autoencoder(train_path, test_path=None, n_features=15, batch_size=1024):
    """Get data optimized for autoencoder training"""
    
    if test_path is None:
        # Single file mode
        df, _ = preprocess_for_any_model(
            train_path, test_path=None, model_type='autoencoder', n_features=n_features
        )
        
        # Extract only normal samples
        normal_df = df[df['label'] == 0]
        X_train_normal = normal_df.drop('label', axis=1)
        
        # Use remaining for test
        X_test = df.drop('label', axis=1)
        y_test = df['label']
    else:
        # Train/test files provided
        X_train_normal, X_test, y_