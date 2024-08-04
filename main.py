from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_and_evaluate_models
from src.models.predict_model import predict_with_model, predict_proba_with_model
from src.visualization.visualize import plot_loan_status, plot_loan_amount_distribution

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('data/raw/credit.csv')

    # Visualize data
    plot_loan_status(df)
    plot_loan_amount_distribution(df)

    # Build features
    x, y = build_features(df)

    # Train models and evaluate
    results, lr_model, rf_model, xtest_scaled = train_and_evaluate_models(x, y)
    
    # Predict using Logistic Regression
    lr_predictions = predict_with_model(lr_model, xtest_scaled)
    lr_predictions_threshold = predict_proba_with_model(lr_model, xtest_scaled, threshold=0.7)

    # Predict using Random Forest
    rf_predictions = predict_with_model(rf_model, xtest_scaled)

    # Print model evaluation results
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"Cross-Validation Mean Accuracy: {metrics['cv_mean_accuracy']}")
        print(f"Cross-Validation Std Dev: {metrics['cv_std']}")
        print("\n")

    # Example output of predictions
    print(f"Logistic Regression Predictions: {lr_predictions[:5]}")
    print(f"Logistic Regression (70% Threshold) Predictions: {lr_predictions_threshold[:5]}")
    print(f"Random Forest Predictions: {rf_predictions[:5]}")

if __name__ == "__main__":
    main()