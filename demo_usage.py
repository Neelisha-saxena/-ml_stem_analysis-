#!/usr/bin/env python3
"""
Demonstration script for the improved STEM funding prediction pipeline.

This script shows how to use the improved ML pipeline in different scenarios.
"""

from improved_ml_pipeline import STEMFundingPredictor, ModelConfig
import pandas as pd
import numpy as np
from pathlib import Path

def demo_basic_usage():
    """Demonstrate basic usage of the improved pipeline."""
    print("="*60)
    print("DEMO: Basic Usage of Improved ML Pipeline")
    print("="*60)
    
    # Create configuration
    config = ModelConfig(
        test_size=0.3,
        random_state=42,
        cv_folds=3,  # Reduced for demo speed
        scoring='roc_auc'
    )
    
    # Initialize predictor
    predictor = STEMFundingPredictor(config)
    
    # Try to load data (will handle missing files gracefully)
    try:
        # First try the processed CSV
        predictor.load_and_prepare_data("data/ml_stem_analysis.csv", file_type='csv')
        print("✅ Successfully loaded processed CSV data")
    except FileNotFoundError:
        try:
            # Fall back to original Excel file
            predictor.load_and_prepare_data(
                "data/2010 Federal STEM Education Inventory Data Set.xls", 
                file_type='excel'
            )
            print("✅ Successfully loaded and processed Excel data")
        except FileNotFoundError:
            print("❌ Data files not found. Please ensure data files are in the data/ directory")
            return None
    
    print(f"📊 Data shape: {predictor.processed_data.shape}")
    print(f"🎯 Target distribution:")
    print(predictor.processed_data['target'].value_counts())
    
    return predictor

def demo_custom_configuration():
    """Demonstrate custom configuration options."""
    print("\n" + "="*60)
    print("DEMO: Custom Configuration")
    print("="*60)
    
    # Custom configuration for faster training
    config = ModelConfig(
        test_size=0.2,
        random_state=123,
        cv_folds=3,
        scoring='roc_auc',
        xgb_params={
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'n_estimators': 100,  # Reduced for demo
            'learning_rate': 0.1,
            'random_state': 123,
            'n_jobs': -1
        }
    )
    
    print("🔧 Custom configuration created:")
    print(f"   - Test size: {config.test_size}")
    print(f"   - CV folds: {config.cv_folds}")
    print(f"   - XGBoost estimators: {config.xgb_params['n_estimators']}")
    
    return config

def demo_model_training_and_evaluation(predictor):
    """Demonstrate model training and evaluation."""
    if predictor is None:
        print("❌ Cannot train model - no data loaded")
        return
    
    print("\n" + "="*60)
    print("DEMO: Model Training and Evaluation")
    print("="*60)
    
    try:
        print("🚀 Starting model training...")
        predictor.train_model()
        
        print("📈 Evaluating model...")
        results = predictor.evaluate_model(show_plots=False)  # Disable plots for demo
        
        print("\n🎉 Training completed successfully!")
        print(f"   Best CV ROC-AUC: {results['best_score']:.4f}")
        print(f"   Test ROC-AUC: {results['test_roc_auc']:.4f}")
        
        # Save model
        predictor.save_model("demo_model.joblib")
        print("💾 Model saved as 'demo_model.joblib'")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return None

def demo_predictions(predictor):
    """Demonstrate making predictions with the trained model."""
    if predictor is None or predictor.trainer.best_model is None:
        print("❌ Cannot make predictions - no trained model")
        return
    
    print("\n" + "="*60)
    print("DEMO: Making Predictions")
    print("="*60)
    
    # Get some sample data for prediction
    X_sample = predictor.processed_data.drop('target', axis=1).head(5)
    
    try:
        # Make predictions
        predictions = predictor.predict(X_sample)
        probabilities = predictor.predict_proba(X_sample)
        
        print("🔮 Sample Predictions:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"   Sample {i+1}: Prediction = {pred}, Probability = {prob[1]:.3f}")
    
    except Exception as e:
        print(f"❌ Error making predictions: {str(e)}")

def create_sample_data():
    """Create sample data for demonstration if real data is not available."""
    print("\n" + "="*60)
    print("DEMO: Creating Sample Data")
    print("="*60)
    
    # Create synthetic data similar to STEM funding data
    np.random.seed(42)
    n_samples = 100
    
    # Create sample features
    data = {
        'C1) Funding FY2008': np.random.exponential(100000, n_samples),
        'C2) Funding FY2009': np.random.exponential(95000, n_samples),
        'Agency_Code': np.random.randint(1, 10, n_samples),
        'Program_Type': np.random.randint(1, 5, n_samples),
        'STEM_Focus_1': np.random.randint(0, 2, n_samples),
        'STEM_Focus_2': np.random.randint(0, 2, n_samples),
        'Geographic_Scope': np.random.randint(1, 4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    # Save sample data
    df.to_csv("data/sample_stem_data.csv", index=False)
    print("📄 Sample data created and saved as 'data/sample_stem_data.csv'")
    
    return df

def main():
    """Main demonstration function."""
    print("🎯 STEM Funding Prediction - Improved Pipeline Demo")
    print("="*60)
    
    # Check if data exists, create sample if not
    data_exists = (
        Path("data/ml_stem_analysis.csv").exists() or 
        Path("data/2010 Federal STEM Education Inventory Data Set.xls").exists()
    )
    
    if not data_exists:
        print("📄 Real data not found, creating sample data for demonstration...")
        create_sample_data()
    
    # Demo 1: Basic usage
    predictor = demo_basic_usage()
    
    # If no real data, use sample data
    if predictor is None:
        try:
            predictor = STEMFundingPredictor()
            predictor.load_and_prepare_data("data/sample_stem_data.csv", file_type='csv')
            print("✅ Using sample data for demonstration")
        except Exception as e:
            print(f"❌ Failed to load sample data: {str(e)}")
            return
    
    # Demo 2: Custom configuration
    custom_config = demo_custom_configuration()
    
    # Demo 3: Model training and evaluation
    results = demo_model_training_and_evaluation(predictor)
    
    # Demo 4: Making predictions
    demo_predictions(predictor)
    
    print("\n" + "="*60)
    print("✨ Demo completed! Check the generated files:")
    print("   - demo_model.joblib (trained model)")
    print("   - code_improvements_summary.md (improvement details)")
    print("   - improved_ml_pipeline.py (improved code)")
    print("="*60)

if __name__ == "__main__":
    main()