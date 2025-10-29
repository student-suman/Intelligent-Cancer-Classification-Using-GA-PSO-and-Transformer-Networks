"""
Data Preprocessing Module for Cancer Gene Expression Analysis

This module handles:
- Loading gene expression datasets (CSV format)
- Missing value imputation
- Z-score normalization
- Train/test split with stratification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for gene expression data.
    
    Attributes:
        scaler: StandardScaler for Z-score normalization
        imputer: SimpleImputer for handling missing values
        feature_names: List of gene/feature names
        n_features: Total number of features in dataset
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = None
        self.n_features = None
        self.random_state = random_state
        
    def load_data(self, file_path, label_column='label'):
        """
        Load gene expression data from CSV file.
        
        Args:
            file_path (str): Path to CSV file
            label_column (str): Name of the column containing class labels
            
        Returns:
            tuple: (features DataFrame, labels Series)
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Separate features and labels
        if label_column in df.columns:
            X = df.drop(columns=[label_column])
            y = df[label_column]
        else:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        self.n_features = len(self.feature_names)
        
        print(f"Loaded {len(X)} samples with {self.n_features} genes")
        print(f"Class distribution:\n{y.value_counts()}")
        
        return X, y
    
    def handle_missing_values(self, X, strategy='mean'):
        """
        Handle missing values using imputation.
        
        Args:
            X (DataFrame or array): Feature matrix
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
            
        Returns:
            array: Imputed feature matrix
        """
        print(f"Handling missing values using {strategy} imputation...")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Check for missing values
        n_missing = np.isnan(X_array).sum()
        if n_missing > 0:
            print(f"Found {n_missing} missing values ({n_missing / X_array.size * 100:.2f}%)")
            self.imputer = SimpleImputer(strategy=strategy)
            X_imputed = self.imputer.fit_transform(X_array)
            print("Missing values imputed successfully")
        else:
            print("No missing values found")
            X_imputed = X_array
            
        return X_imputed
    
    def normalize_zscore(self, X_train, X_test=None):
        """
        Apply Z-score normalization (standardization).
        
        Each feature is scaled to have mean=0 and std=1.
        
        Args:
            X_train (array): Training feature matrix
            X_test (array, optional): Test feature matrix
            
        Returns:
            tuple or array: Normalized training (and test if provided) data
        """
        print("Applying Z-score normalization...")
        
        # Fit on training data
        X_train_normalized = self.scaler.fit_transform(X_train)
        print(f"Train data normalized - Mean: {X_train_normalized.mean():.4f}, Std: {X_train_normalized.std():.4f}")
        
        if X_test is not None:
            # Transform test data using training statistics
            X_test_normalized = self.scaler.transform(X_test)
            print(f"Test data normalized - Mean: {X_test_normalized.mean():.4f}, Std: {X_test_normalized.std():.4f}")
            return X_train_normalized, X_test_normalized
        
        return X_train_normalized
    
    def split_data(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets.
        
        Args:
            X (array): Feature matrix
            y (array): Labels
            test_size (float): Proportion of data for testing
            stratify (bool): Whether to stratify split by class labels
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print(f"Splitting data (test_size={test_size})...")
        
        stratify_labels = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=stratify_labels,
            random_state=self.random_state
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path, label_column='label', test_size=0.2):
        """
        Complete preprocessing pipeline.
        
        Executes the following steps:
        1. Load data from CSV
        2. Handle missing values
        3. Split into train/test
        4. Apply Z-score normalization
        
        Args:
            file_path (str): Path to CSV file
            label_column (str): Name of label column
            test_size (float): Proportion for test set
            
        Returns:
            dict: Dictionary containing all preprocessed data and metadata
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load data
        X, y = self.load_data(file_path, label_column)
        
        # Step 2: Handle missing values
        X_clean = self.handle_missing_values(X)
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = self.split_data(
            X_clean, y.values, test_size=test_size
        )
        
        # Step 4: Normalize
        X_train_norm, X_test_norm = self.normalize_zscore(X_train, X_test)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
        # Return comprehensive data dictionary
        return {
            'X_train': X_train_norm,
            'X_test': X_test_norm,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'n_classes': len(np.unique(y))
        }


def generate_synthetic_dataset(n_samples=1000, n_genes=20000, n_classes=5, 
                                n_informative=200, output_file='synthetic_cancer_data.csv',
                                random_state=42):
    """
    Generate synthetic gene expression dataset for testing.
    
    Simulates cancer gene expression data with:
    - High dimensionality (many genes)
    - Multiple cancer types (classes)
    - Realistic gene expression patterns
    
    Args:
        n_samples (int): Number of samples
        n_genes (int): Number of genes/features
        n_classes (int): Number of cancer types
        n_informative (int): Number of informative genes
        output_file (str): Output CSV filename
        random_state (int): Random seed
        
    Returns:
        str: Path to generated CSV file
    """
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC CANCER GENE EXPRESSION DATASET")
    print("="*60 + "\n")
    
    np.random.seed(random_state)
    
    # Generate base gene expression (log-normal distribution typical for RNA-seq)
    X = np.random.lognormal(mean=5, sigma=2, size=(n_samples, n_genes))
    
    # Generate class labels
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Add class-specific patterns to informative genes
    informative_genes = np.random.choice(n_genes, n_informative, replace=False)
    
    for gene_idx in informative_genes:
        for class_label in range(n_classes):
            # Samples of this class get differential expression
            class_mask = (y == class_label)
            # Add class-specific signal
            X[class_mask, gene_idx] *= (1 + 0.5 * class_label + np.random.normal(0, 0.2, class_mask.sum()))
    
    # Add some missing values (5%)
    missing_mask = np.random.random(X.shape) < 0.05
    X[missing_mask] = np.nan
    
    # Create DataFrame
    gene_names = [f'Gene_{i:05d}' for i in range(n_genes)]
    df = pd.DataFrame(X, columns=gene_names)
    df['label'] = y
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"  Samples: {n_samples}")
    print(f"  Genes: {n_genes}")
    print(f"  Classes: {n_classes}")
    print(f"  Informative genes: {n_informative}")
    print(f"  Missing values: {missing_mask.sum()} ({missing_mask.sum() / X.size * 100:.2f}%)")
    print(f"  Saved to: {output_file}")
    print(f"\nClass distribution:")
    for i in range(n_classes):
        count = (y == i).sum()
        print(f"  Class {i}: {count} samples ({count/n_samples*100:.1f}%)")
    
    return output_file


if __name__ == "__main__":
    # Example usage: Generate synthetic dataset
    dataset_path = generate_synthetic_dataset(
        n_samples=1000,
        n_genes=20000,
        n_classes=5,
        n_informative=200
    )
    
    # Example usage: Preprocess the dataset
    preprocessor = DataPreprocessor(random_state=42)
    data = preprocessor.preprocess_pipeline(dataset_path, test_size=0.2)
    
    print("\nPreprocessed data shapes:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_test: {data['X_test'].shape}")
    print(f"  y_train: {data['y_train'].shape}")
    print(f"  y_test: {data['y_test'].shape}")
