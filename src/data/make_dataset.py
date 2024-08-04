import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Impute missing values without using inplace
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Drop 'Loan_ID' as it's not needed
    df = df.drop('Loan_ID', axis=1)
    
    return df