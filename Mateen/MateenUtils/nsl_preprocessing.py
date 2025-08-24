"""
Universal NSL-KDD Preprocessing Module
=====================================

Enhanced NSL-KDD preprocessing that works with ANY model including Mateen
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class UniversalNSLKDDPreprocessor:
    def __init__(self, n_features=15, sampling_strategy='smote_tomek', scaler_type='robust'):
        self.n_features = n_features
        self.sampling_strategy = sampling_strategy
        self.scaler_type = scaler_type
        self.label_encoders = {}
        self.scaler = None
        self.selected_features = None

    def load_and_clean_data(self, train_path, test_path):
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

        train_df = pd.read_csv(train_path, names=columns)
        test_df = pd.read_csv(test_path, names=columns)

        drop_cols = ['difficulty_level', 'num_outbound_cmds', 'is_host_login']
        train_df = train_df.drop(drop_cols, axis=1)
        test_df = test_df.drop(drop_cols, axis=1)

        train_df['label'] = train_df['label'].apply(lambda x: 0 if str(x).strip() == 'normal' else 1)
        test_df['label'] = test_df['label'].apply(lambda x: 0 if str(x).strip() == 'normal' else 1)

        return train_df, test_df

    def create_engineered_features(self, df):
        df_eng = df.copy()

        df_eng['bytes_ratio'] = np.where(df_eng['dst_bytes'] > 0,
                                       df_eng['src_bytes'] / df_eng['dst_bytes'], 0)
        df_eng['total_bytes'] = df_eng['src_bytes'] + df_eng['dst_bytes']
        df_eng['bytes_per_second'] = np.where(df_eng['duration'] > 0,
                                            df_eng['total_bytes'] / df_eng['duration'], 0)
        df_eng['srv_ratio'] = np.where(df_eng['count'] > 0,
                                     df_eng['srv_count'] / df_eng['count'], 0)
        df_eng['error_ratio'] = df_eng['serror_rate'] + df_eng['rerror_rate']
        df_eng['srv_error_ratio'] = df_eng['srv_serror_rate'] + df_eng['srv_rerror_rate']
        df_eng['host_srv_ratio'] = np.where(df_eng['dst_host_count'] > 0,
                                          df_eng['dst_host_srv_count'] / df_eng['dst_host_count'], 0)
        df_eng['host_error_total'] = (df_eng['dst_host_serror_rate'] +
                                    df_eng['dst_host_rerror_rate'])
        df_eng['security_score'] = (df_eng['num_failed_logins'] +
                                  df_eng['num_compromised'] +
                                  df_eng['num_root'])
        df_eng['file_activity'] = df_eng['num_file_creations'] + df_eng['num_shells']

        log_features = ['src_bytes', 'dst_bytes', 'count', 'srv_count', 'dst_host_count']
        for feature in log_features:
            if feature in df_eng.columns:
                df_eng[f'{feature}_log'] = np.log1p(df_eng[feature])

        return df_eng

    def handle_outliers(self, X_train, X_test, method='iqr'):
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()

        numeric_cols = X_train_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = X_train_clean[col].quantile(0.25)
            Q3 = X_train_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            X_train_clean[col] = X_train_clean[col].clip(lower_bound, upper_bound)
            X_test_clean[col] = X_test_clean[col].clip(lower_bound, upper_bound)

        return X_train_clean, X_test_clean

    def encode_categorical_features(self, X_train, X_test):
        categorical_columns = ['protocol_type', 'service', 'flag']
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        for col in categorical_columns:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            self.label_encoders[col] = le

            test_values = X_test[col].astype(str)
            for val in test_values.unique():
                if val not in le.classes_:
                    test_values = test_values.replace(val, le.classes_[0])
            X_test_encoded[col] = le.transform(test_values)

            freq_map = X_train[col].value_counts().to_dict()
            X_train_encoded[f'{col}_freq'] = X_train[col].map(freq_map)
            X_test_encoded[f'{col}_freq'] = X_test[col].map(freq_map).fillna(0)

        return X_train_encoded, X_test_encoded

    def scale_features(self, X_train, X_test):
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])

        return X_train_scaled, X_test_scaled

    def consensus_feature_selection(self, X_train, y_train, X_test):
        xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        xgb_scores = xgb.feature_importances_

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_scores = rf.feature_importances_

        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        f_scores = f_classif(X_train, y_train)[0]

        xgb_norm = xgb_scores / np.max(xgb_scores)
        rf_norm = rf_scores / np.max(rf_scores)
        mi_norm = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
        f_norm = f_scores / np.max(f_scores)

        combined_scores = 0.25 * xgb_norm + 0.25 * rf_norm + 0.25 * mi_norm + 0.25 * f_norm

        feature_indices = np.argsort(combined_scores)[-self.n_features:]
        self.selected_features = X_train.columns[feature_indices].tolist()

        print(f"Selected {len(self.selected_features)} features using consensus:")
        for i, feature in enumerate(self.selected_features, 1):
            print(f"  {i:2d}. {feature}")

        return X_train.iloc[:, feature_indices], X_test.iloc[:, feature_indices]

    def fit_transform(self, train_path, test_path):
        print("=== Universal NSL-KDD Preprocessing ===")

        print("1. Loading and cleaning data...")
        train_df, test_df = self.load_and_clean_data(train_path, test_path)

        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']

        print(f"   Original: Train {X_train.shape}, Test {X_test.shape}")

        print("2. Engineering features...")
        X_train = self.create_engineered_features(X_train)
        X_test = self.create_engineered_features(X_test)
        print(f"   After engineering: {X_train.shape}")

        print("3. Handling outliers...")
        X_train, X_test = self.handle_outliers(X_train, X_test)

        print("4. Encoding categorical features...")
        X_train, X_test = self.encode_categorical_features(X_train, X_test)

        print("5. Scaling features...")
        X_train, X_test = self.scale_features(X_train, X_test)

        print(f"6. Selecting top {self.n_features} features...")
        X_train, X_test = self.consensus_feature_selection(X_train, y_train, X_test)

        print(f"   Final: Train {X_train.shape}, Test {X_test.shape}")
        print(f"   Class balance: Normal {np.sum(y_train==0)}, Attack {np.sum(y_train==1)}")

        return X_train, y_train, X_test, y_test

def preprocess_for_any_model(train_path, test_path, model_type='autoencoder', n_features=15, **kwargs):
    preprocessor = UniversalNSLKDDPreprocessor(
        n_features=n_features,
        sampling_strategy='none',
        scaler_type='robust'
    )

    X_train, y_train, X_test, y_test = preprocessor.fit_transform(train_path, test_path)

    if model_type == 'autoencoder':
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask]
        return X_train_normal, X_test, y_test, preprocessor.selected_features, preprocessor
    else:
        return X_train, y_train, X_test, y_test, preprocessor.selected_features, preprocessor

def prepare_datasets(x_train, y_train):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if hasattr(x_train, 'values'):
        x_train = x_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    X_tensor = torch.FloatTensor(x_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    benign_train = x_train[y_train == 0] if y_train is not None else x_train
    return train_loader, benign_train

def loading_datasets(x_data):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if hasattr(x_data, 'values'):
        x_data = x_data.values

    X_tensor = torch.FloatTensor(x_data)
    dataset = TensorDataset(X_tensor, X_tensor)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    return data_loader, None

def prepare_data(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], str):
        scenario = args[0]
        if scenario in ["NSLKDD", "NSL-KDD"]:
            train_path = "/content/drive/MyDrive/Colab Notebooks/data/KDDTrain+.txt"
            test_path = "/content/drive/MyDrive/Colab Notebooks/data/KDDTest+.txt"

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

    elif len(args) == 1 and hasattr(args[0], 'shape'):
        return loading_datasets(args[0])

    elif len(args) == 2:
        return prepare_datasets(args[0], args[1])

    else:
        raise ValueError(f"Unsupported prepare_data call")

def partition_array(x_data, y_data, slice_size):
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
