#!/usr/bin/env python3
"""
Improved ML Pipeline for STEM Education Funding Growth Prediction

This module provides a complete, production-ready pipeline for predicting
funding growth in STEM education programs using XGBoost with proper
data preprocessing, feature engineering, and model evaluation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import warnings
from dataclasses import dataclass

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Imbalanced learning
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class ModelConfig:
    """Configuration class for model parameters and settings."""
    test_size: float = 0.3
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1
    scoring: str = 'roc_auc'
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }


class DataLoader:
    """Handle data loading and initial validation."""
    
    @staticmethod
    def load_excel_data(file_path: str) -> pd.DataFrame:
        """
        Load Excel data with proper error handling.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in ['.xls', '.xlsx']:
            raise ValueError(f"Invalid file format. Expected .xls or .xlsx, got {path.suffix}")
        
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_excel(file_path, header=1)
            logger.info(f"Successfully loaded {len(data)} rows and {len(data.columns)} columns")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    @staticmethod
    def load_csv_data(file_path: str) -> pd.DataFrame:
        """Load CSV data with validation."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() != '.csv':
            raise ValueError(f"Invalid file format. Expected .csv, got {path.suffix}")
        
        try:
            logger.info(f"Loading CSV data from {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(data)} rows and {len(data.columns)} columns")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise


class DataPreprocessor:
    """Handle all data preprocessing tasks."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.label_encoders = {}
        self.feature_names = None
    
    def create_target_variable(self, data: pd.DataFrame, 
                             funding_2008_col: str = 'C1) Funding FY2008',
                             funding_2009_col: str = 'C2) Funding FY2009') -> pd.DataFrame:
        """
        Create target variable based on funding growth.
        
        Args:
            data: Input DataFrame
            funding_2008_col: Column name for 2008 funding
            funding_2009_col: Column name for 2009 funding
            
        Returns:
            DataFrame with 'Growth' and 'target' columns added
        """
        data_copy = data.copy()
        
        # Fill NaN values with 0 for funding columns
        data_copy[funding_2008_col] = data_copy[funding_2008_col].fillna(0)
        data_copy[funding_2009_col] = data_copy[funding_2009_col].fillna(0)
        
        # Vectorized calculation of growth percentage
        funding_2008 = data_copy[funding_2008_col]
        funding_2009 = data_copy[funding_2009_col]
        
        # Handle division by zero cases
        growth = np.where(
            funding_2008 == 0,
            np.where(
                funding_2009 == 0, 0,
                np.where(funding_2009 < 0, -100, 100)
            ),
            ((funding_2009 - funding_2008) / funding_2008) * 100
        )
        
        data_copy['Growth'] = growth
        data_copy['target'] = (growth >= 0).astype(int)
        
        logger.info(f"Created target variable. Class distribution:")
        logger.info(f"Positive growth (1): {(data_copy['target'] == 1).sum()}")
        logger.info(f"Negative growth (0): {(data_copy['target'] == 0).sum()}")
        
        return data_copy
    
    def clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names."""
        data_copy = data.copy()
        
        # Fix unnamed columns
        j = ""
        k = 1
        for i, col in enumerate(data_copy.columns):
            if col.startswith("Unnamed"):
                new_name = f"{j} {k}"
                data_copy.rename(columns={col: new_name}, inplace=True)
                k += 1
            else:
                j = col
                k = 1
        
        return data_copy
    
    def identify_columns_to_drop(self, data: pd.DataFrame, 
                               manual_drops: Optional[List[str]] = None) -> List[str]:
        """
        Identify columns to drop based on various criteria.
        
        Args:
            data: Input DataFrame
            manual_drops: List of manually specified columns to drop
            
        Returns:
            List of column names to drop
        """
        if manual_drops is None:
            manual_drops = ['Growth', "N) STEM Discipline Focus 12"]
        
        cols_to_drop = manual_drops.copy()
        n_rows = len(data)
        
        for col in data.columns:
            # Drop columns with too many unique values (likely IDs)
            unique_count = data[col].nunique()
            if unique_count in [n_rows, n_rows - 1]:
                cols_to_drop.append(col)
                continue
            
            # Drop columns with only one unique value
            if unique_count == 1:
                cols_to_drop.append(col)
                continue
            
            # Drop columns with too many missing values (>60%)
            missing_ratio = data[col].isnull().sum() / n_rows
            if missing_ratio > 0.6:
                cols_to_drop.append(col)
                continue
        
        # Remove duplicates and ensure columns exist
        cols_to_drop = list(set(cols_to_drop))
        existing_cols = [col for col in cols_to_drop if col in data.columns]
        
        logger.info(f"Identified {len(existing_cols)} columns to drop")
        return existing_cols
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        data_copy = data.copy()
        
        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                # Create label encoder for this column
                le = LabelEncoder()
                
                # Handle NaN values by creating a mask
                not_null_mask = data_copy[col].notna()
                
                if not_null_mask.any():
                    # Fit encoder on non-null values
                    data_copy.loc[not_null_mask, col] = le.fit_transform(
                        data_copy.loc[not_null_mask, col]
                    )
                    self.label_encoders[col] = le
                
                # Convert to numeric, keeping NaN as NaN
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
        
        return data_copy
    
    def create_preprocessing_pipeline(self, data: pd.DataFrame) -> Pipeline:
        """
        Create a comprehensive preprocessing pipeline.
        
        Args:
            data: Input DataFrame to determine column types
            
        Returns:
            Sklearn Pipeline for preprocessing
        """
        # Identify numeric and categorical columns
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='drop'
        )
        
        self.feature_names = numeric_features
        logger.info(f"Created preprocessing pipeline for {len(numeric_features)} numeric features")
        
        return preprocessor


class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.best_model = None
        self.preprocessor = None
        self.results = {}
    
    def create_model_pipeline(self, preprocessor: Pipeline) -> ImbPipeline:
        """
        Create complete model pipeline with preprocessing and resampling.
        
        Args:
            preprocessor: Preprocessing pipeline
            
        Returns:
            Complete model pipeline
        """
        # Create XGBoost classifier
        xgb_classifier = xgb.XGBClassifier(**self.config.xgb_params)
        
        # Create pipeline with preprocessing, resampling, and classification
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('resampler', SMOTETomek(random_state=self.config.random_state)),
            ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
            ('classifier', xgb_classifier)
        ])
        
        return pipeline
    
    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """Get hyperparameter grid for GridSearchCV."""
        return {
            'classifier__n_estimators': [100, 300, 500],
            'classifier__max_depth': [3, 4, 5, 6],
            'classifier__min_child_weight': [1, 3, 5],
            'classifier__gamma': [0, 0.1, 0.2],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
            'classifier__learning_rate': [0.01, 0.02, 0.1],
            'feature_selection__k': [10, 20, 30, 'all']
        }
    
    def train_with_grid_search(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Starting model training with grid search...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        # Create preprocessor
        preprocessor = DataPreprocessor(self.config).create_preprocessing_pipeline(X_train)
        
        # Create pipeline
        pipeline = self.create_model_pipeline(preprocessor)
        
        # Setup grid search
        param_grid = self.get_hyperparameter_grid()
        
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds, 
            shuffle=True, 
            random_state=self.config.random_state
        )
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs,
            verbose=1
        )
        
        # Fit grid search
        logger.info("Fitting grid search...")
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.best_model = grid_search.best_estimator_
        self.preprocessor = preprocessor
        
        # Evaluate on test set
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        self.results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': self.best_model.score(X_test, y_test),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"Training completed!")
        logger.info(f"Best CV score: {self.results['best_score']:.4f}")
        logger.info(f"Test ROC-AUC: {self.results['test_roc_auc']:.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_with_grid_search first.")
        
        # Get feature importance from XGBoost
        importance = self.best_model.named_steps['classifier'].feature_importances_
        
        # Get feature names (this is simplified - you might need to adjust based on your pipeline)
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


class ModelEvaluator:
    """Handle model evaluation and visualization."""
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      title: str = "ROC Curve") -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> None:
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20) -> None:
        """Plot feature importance."""
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()


class STEMFundingPredictor:
    """Main class orchestrating the entire ML pipeline."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor(self.config)
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator()
        self.raw_data = None
        self.processed_data = None
    
    def load_and_prepare_data(self, file_path: str, file_type: str = 'excel') -> None:
        """
        Load and prepare data for training.
        
        Args:
            file_path: Path to data file
            file_type: Type of file ('excel' or 'csv')
        """
        # Load data
        if file_type.lower() == 'excel':
            self.raw_data = self.data_loader.load_excel_data(file_path)
        elif file_type.lower() == 'csv':
            self.raw_data = self.data_loader.load_csv_data(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Clean column names
        self.raw_data = self.preprocessor.clean_column_names(self.raw_data)
        
        # Create target variable (if not already present)
        if 'target' not in self.raw_data.columns:
            self.raw_data = self.preprocessor.create_target_variable(self.raw_data)
        
        # Identify and drop unnecessary columns
        cols_to_drop = self.preprocessor.identify_columns_to_drop(self.raw_data)
        self.processed_data = self.raw_data.drop(columns=cols_to_drop, errors='ignore')
        
        # Encode categorical features
        self.processed_data = self.preprocessor.encode_categorical_features(self.processed_data)
        
        logger.info(f"Data preparation completed. Shape: {self.processed_data.shape}")
    
    def train_model(self) -> None:
        """Train the model."""
        if self.processed_data is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data first.")
        
        # Separate features and target
        X = self.processed_data.drop('target', axis=1)
        y = self.processed_data['target']
        
        # Train model
        self.trainer.train_with_grid_search(X, y)
    
    def evaluate_model(self, show_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            show_plots: Whether to show evaluation plots
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.trainer.best_model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        results = self.trainer.results
        
        # Print results
        print("="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Best CV ROC-AUC Score: {results['best_score']:.4f}")
        print(f"Test ROC-AUC Score: {results['test_roc_auc']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print("\nBest Parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print("\nClassification Report:")
        print(results['classification_report'])
        
        if show_plots:
            # Plot ROC curve
            self.evaluator.plot_roc_curve(
                results['y_test'], 
                results['y_pred_proba']
            )
            
            # Plot Precision-Recall curve
            self.evaluator.plot_precision_recall_curve(
                results['y_test'], 
                results['y_pred_proba']
            )
            
            # Plot confusion matrix
            self.evaluator.plot_confusion_matrix(results['confusion_matrix'])
            
            # Plot feature importance
            try:
                importance_df = self.trainer.get_feature_importance()
                self.evaluator.plot_feature_importance(importance_df)
            except Exception as e:
                logger.warning(f"Could not plot feature importance: {str(e)}")
        
        return results
    
    def save_model(self, file_path: str) -> None:
        """Save the trained model."""
        if self.trainer.best_model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        import joblib
        joblib.dump(self.trainer.best_model, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.trainer.best_model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        return self.trainer.best_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.trainer.best_model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        return self.trainer.best_model.predict_proba(X)


def main():
    """Main execution function."""
    # Configuration
    config = ModelConfig(
        test_size=0.3,
        random_state=42,
        cv_folds=5,
        scoring='roc_auc'
    )
    
    # Initialize predictor
    predictor = STEMFundingPredictor(config)
    
    # Check if processed data already exists
    try:
        predictor.load_and_prepare_data("data/ml_stem_analysis.csv", file_type='csv')
    except FileNotFoundError:
        # If not, load from Excel and process
        logger.info("CSV file not found, loading from Excel...")
        predictor.load_and_prepare_data(
            "data/2010 Federal STEM Education Inventory Data Set.xls", 
            file_type='excel'
        )
        # Save processed data
        predictor.processed_data.to_csv("data/ml_stem_analysis_improved.csv", index=False)
    
    # Train model
    predictor.train_model()
    
    # Evaluate model
    results = predictor.evaluate_model(show_plots=True)
    
    # Save model
    predictor.save_model("improved_stem_model.joblib")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()