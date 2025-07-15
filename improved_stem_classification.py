"""
STEM Education Classification Project - Improved Version
======================================================

This script contains an improved implementation of the imbalanced classification task 
on the 2010 Federal STEM Education Inventory dataset using XGBoost.

Key Improvements:
- Better code organization and structure
- Comprehensive documentation and docstrings
- Robust error handling and validation
- Modern best practices and updated dependencies
- Improved data preprocessing pipeline
- Better hyperparameter optimization
- Comprehensive evaluation metrics
- Modular and reusable functions

Author: Improved by AI Assistant
Target ROC AUC Score: 0.9145+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import List, Tuple, Dict, Any, Optional
import logging

# Machine Learning imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, roc_curve, auc, classification_report, 
                           confusion_matrix, precision_recall_curve, accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
import xgboost as xgb
from xgboost import XGBClassifier

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Enhanced data loading class with robust error handling and validation."""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file with comprehensive validation.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        
        # Validate file existence
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        if path.suffix.lower() != '.csv':
            raise ValueError(f"Unsupported file format: {path.suffix}. Only CSV files are supported.")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            logger.info(f"Dataset shape: {data.shape}")
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")


class DataPreprocessor:
    """Enhanced data preprocessing class with improved functionality."""
    
    def __init__(self, missing_threshold: float = 0.6, unique_threshold: int = 250):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            missing_threshold (float): Threshold for dropping columns with too many missing values
            unique_threshold (int): Threshold for dropping columns with too many unique values
        """
        self.missing_threshold = missing_threshold
        self.unique_threshold = unique_threshold
        self.columns_to_drop = []
        self.label_encoders = {}
        
    def identify_columns_to_drop(self, data: pd.DataFrame, 
                               manual_drops: Optional[List[str]] = None) -> List[str]:
        """
        Identify columns to drop based on various criteria.
        
        Args:
            data (pd.DataFrame): Input dataset
            manual_drops (List[str], optional): Manually specified columns to drop
            
        Returns:
            List[str]: List of column names to drop
        """
        columns_to_drop = []
        
        # Add manually specified columns
        if manual_drops:
            for col in manual_drops:
                if col in data.columns:
                    columns_to_drop.append(col)
                    logger.info(f"Added manual drop column: {col}")
        
        # Drop columns with too many unique values (likely IDs or unique identifiers)
        for col in data.columns:
            unique_count = data[col].nunique()
            if unique_count >= self.unique_threshold:
                columns_to_drop.append(col)
                logger.info(f"Dropping {col} - too many unique values: {unique_count}")
        
        # Drop columns with too many missing values
        missing_threshold_count = int(len(data) * self.missing_threshold)
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > missing_threshold_count:
                columns_to_drop.append(col)
                logger.info(f"Dropping {col} - too many missing values: {missing_count}/{len(data)}")
        
        # Remove duplicates
        columns_to_drop = list(set(columns_to_drop))
        
        # Validate columns exist in data
        valid_columns = [col for col in columns_to_drop if col in data.columns]
        invalid_columns = [col for col in columns_to_drop if col not in data.columns]
        
        if invalid_columns:
            logger.warning(f"Invalid columns to drop (not in dataset): {invalid_columns}")
        
        self.columns_to_drop = valid_columns
        return valid_columns
    
    def drop_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drop specified columns from the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            columns (List[str]): Columns to drop
            
        Returns:
            pd.DataFrame: Dataset with columns removed
        """
        # Validate columns exist
        existing_columns = [col for col in columns if col in data.columns]
        non_existing = [col for col in columns if col not in data.columns]
        
        if non_existing:
            logger.warning(f"Columns not found in dataset: {non_existing}")
        
        if existing_columns:
            data = data.drop(columns=existing_columns)
            logger.info(f"Dropped {len(existing_columns)} columns: {existing_columns}")
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        data_encoded = data.copy()
        
        for col in data_encoded.columns:
            if data_encoded[col].dtype == 'object':
                # Create label encoder for this column
                le = LabelEncoder()
                
                # Handle missing values by filling with a placeholder
                non_null_mask = data_encoded[col].notna()
                if non_null_mask.sum() > 0:  # Only encode if there are non-null values
                    data_encoded.loc[non_null_mask, col] = le.fit_transform(
                        data_encoded.loc[non_null_mask, col]
                    )
                    self.label_encoders[col] = le
                    logger.info(f"Encoded categorical column: {col}")
                else:
                    logger.warning(f"Column {col} has no non-null values, filling with 0")
                    data_encoded[col] = 0
        
        return data_encoded
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using mean imputation for numerical features.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        data_imputed = data.copy()
        
        # Use mean imputation for numerical columns
        numerical_columns = data_imputed.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            if data_imputed[col].isnull().sum() > 0:
                mean_value = data_imputed[col].mean()
                data_imputed[col].fillna(mean_value, inplace=True)
                logger.info(f"Imputed missing values in {col} with mean: {mean_value:.2f}")
        
        # For any remaining non-numerical columns with missing values, fill with mode
        for col in data_imputed.columns:
            if data_imputed[col].isnull().sum() > 0:
                mode_value = data_imputed[col].mode()
                if len(mode_value) > 0:
                    data_imputed[col].fillna(mode_value[0], inplace=True)
                    logger.info(f"Imputed missing values in {col} with mode: {mode_value[0]}")
                else:
                    data_imputed[col].fillna(0, inplace=True)
                    logger.info(f"Imputed missing values in {col} with 0 (no mode found)")
        
        return data_imputed
    
    def preprocess_data(self, data: pd.DataFrame, 
                       manual_drops: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Input dataset
            manual_drops (List[str], optional): Manually specified columns to drop
            
        Returns:
            pd.DataFrame: Fully preprocessed dataset
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Step 1: Identify and drop columns
        columns_to_drop = self.identify_columns_to_drop(data, manual_drops)
        data_processed = self.drop_columns(data, columns_to_drop)
        
        # Step 2: Encode categorical features
        data_processed = self.encode_categorical_features(data_processed)
        
        # Step 3: Handle missing values
        data_processed = self.handle_missing_values(data_processed)
        
        logger.info(f"Preprocessing complete. Final shape: {data_processed.shape}")
        return data_processed


class ModelTrainer:
    """Enhanced model training class with hyperparameter optimization."""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the model trainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.cv_results = None
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting features and target.
        
        Args:
            data (pd.DataFrame): Preprocessed dataset
            target_column (str): Name of the target column
            test_size (float): Proportion of test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def apply_smote_tomek(self, X_train: pd.DataFrame, 
                         y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE-Tomek for handling class imbalance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Tuple: Resampled X_train, y_train
        """
        logger.info("Applying SMOTE-Tomek for class imbalance...")
        logger.info(f"Original class distribution:\n{y_train.value_counts()}")
        
        # Updated parameter names for newer versions
        smote_tomek = SMOTETomek(random_state=self.random_state)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled)
        
        logger.info(f"Resampled class distribution:\n{y_resampled.value_counts()}")
        return X_resampled, y_resampled
    
    def hyperparameter_optimization(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  use_randomized: bool = True, n_iter: int = 50) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization for XGBoost.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            use_randomized (bool): Use RandomizedSearchCV instead of GridSearchCV
            n_iter (int): Number of iterations for RandomizedSearchCV
            
        Returns:
            Dict: Best parameters found
        """
        logger.info("Starting hyperparameter optimization...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [500, 600, 700, 800, 900, 1000],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.5, 1.0, 2.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'learning_rate': [0.01, 0.02, 0.05, 0.1]
        }
        
        # Base model
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        # Choose search method
        if use_randomized:
            search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='roc_auc',
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        else:
            search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        self.best_params = search.best_params_
        self.best_model = search.best_estimator_
        self.cv_results = search.cv_results_
        
        logger.info(f"Best ROC AUC: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         best_params: Optional[Dict[str, Any]] = None) -> XGBClassifier:
        """
        Train the final model with best parameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            best_params (Dict, optional): Best parameters from hyperparameter tuning
            
        Returns:
            XGBClassifier: Trained model
        """
        if best_params is None:
            best_params = self.best_params or {}
        
        logger.info("Training final model...")
        
        final_model = XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            **best_params
        )
        
        final_model.fit(X_train, y_train)
        self.best_model = final_model
        
        logger.info("Final model training complete")
        return final_model


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    @staticmethod
    def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        logger.info("Model Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def plot_roc_curve(model: XGBClassifier, X_test: pd.DataFrame, 
                      y_test: pd.Series, save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_path (str, optional): Path to save the plot
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def generate_classification_report(model: XGBClassifier, X_test: pd.DataFrame, 
                                     y_test: pd.Series) -> str:
        """
        Generate detailed classification report.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            str: Classification report
        """
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        logger.info("Classification Report:")
        logger.info(f"\n{report}")
        
        return report


def main():
    """Main execution function."""
    logger.info("Starting STEM Education Classification Project...")
    
    try:
        # 1. Load Data
        data_loader = DataLoader()
        data = data_loader.load_data("data/ml_stem_analysis.csv")
        
        # 2. Data Preprocessing
        preprocessor = DataPreprocessor(missing_threshold=0.6, unique_threshold=250)
        manual_drops = ['Growth', "N) STEM Discipline Focus 12"]  # From original code
        processed_data = preprocessor.preprocess_data(data, manual_drops=manual_drops)
        
        # 3. Model Training
        trainer = ModelTrainer(random_state=RANDOM_STATE)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            processed_data, target_column='target', test_size=0.3
        )
        
        # Apply SMOTE-Tomek
        X_train_resampled, y_train_resampled = trainer.apply_smote_tomek(X_train, y_train)
        
        # Hyperparameter optimization
        best_params = trainer.hyperparameter_optimization(
            X_train_resampled, y_train_resampled, use_randomized=True, n_iter=50
        )
        
        # Train final model
        final_model = trainer.train_final_model(X_train_resampled, y_train_resampled, best_params)
        
        # 4. Model Evaluation
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(final_model, X_test, y_test)
        
        # Generate visualizations and reports
        evaluator.plot_roc_curve(final_model, X_test, y_test, save_path="roc_curve.png")
        evaluator.generate_classification_report(final_model, X_test, y_test)
        
        logger.info(f"🎉 Project completed successfully! ROC AUC: {metrics['roc_auc']:.4f}")
        
        return final_model, metrics
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    model, results = main()