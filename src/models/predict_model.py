def predict_with_model(model, xtest_scaled):
    # Predict the loan eligibility on the testing set
    ypred = model.predict(xtest_scaled)
    
    # Return predictions
    return ypred

def predict_proba_with_model(model, xtest_scaled, threshold=0.5):
    # Get the probability predictions
    proba = model.predict_proba(xtest_scaled)[:, 1]
    
    # Apply the threshold to classify based on probabilities
    ypred_threshold = (proba >= threshold).astype(int)
    
    return ypred_threshold