import seaborn as sns
import matplotlib.pyplot as plt

def plot_loan_status(df):
    df['Loan_Status'].value_counts().plot.bar()
    plt.title('Loan Status Distribution')
    plt.show()

def plot_loan_amount_distribution(df):
    sns.histplot(df['LoanAmount'], kde=True)
    plt.title('Loan Amount Distribution')
    plt.show()