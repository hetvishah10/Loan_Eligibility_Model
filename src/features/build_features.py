import pandas as pd

def build_features(df):
    # Create dummy variables for categorical features
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype="int")
    
    # Separate features and target variable
    x = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    return x, y