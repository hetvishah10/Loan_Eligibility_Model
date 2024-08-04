from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate_models(x, y):
    # Split the data into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Logistic Regression
    lr_model = LogisticRegression().fit(xtrain_scaled, ytrain)
    lr_predictions = lr_model.predict(xtest_scaled)
    lr_accuracy = accuracy_score(ytest, lr_predictions)
    lr_confusion = confusion_matrix(ytest, lr_predictions)

    # Logistic Regression - Cross Validation
    kfold = KFold(n_splits=5)
    lr_cv_scores = cross_val_score(lr_model, xtrain_scaled, ytrain, cv=kfold)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='sqrt')
    rf_model.fit(xtrain, ytrain)
    rf_predictions = rf_model.predict(xtest)
    rf_accuracy = accuracy_score(ytest, rf_predictions)
    rf_confusion = confusion_matrix(ytest, rf_predictions)

    # Random Forest - Cross Validation
    rf_cv_scores = cross_val_score(rf_model, xtrain_scaled, ytrain, cv=kfold)

    results = {
        'LogisticRegression': {
            'accuracy': lr_accuracy,
            'confusion_matrix': lr_confusion,
            'cv_mean_accuracy': lr_cv_scores.mean(),
            'cv_std': lr_cv_scores.std()
        },
        'RandomForest': {
            'accuracy': rf_accuracy,
            'confusion_matrix': rf_confusion,
            'cv_mean_accuracy': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std()
        }
    }

    return results, lr_model, rf_model, xtest_scaled