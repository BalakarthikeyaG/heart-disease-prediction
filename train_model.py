import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

def generate_synthetic_data(num_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Age (29 to 77)
    age = np.random.randint(29, 78, size=num_samples)
    
    # 2. Sex (0 = Female, 1 = Male)
    sex = np.random.choice([0, 1], p=[0.45, 0.55], size=num_samples)
    
    # 3. Chest Pain Type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
    cp = np.random.choice([0, 1, 2, 3], p=[0.40, 0.25, 0.25, 0.10], size=num_samples)
    
    # 4. Resting Blood Pressure (trestbps) (94 to 200)
    trestbps = np.random.normal(loc=130, scale=17, size=num_samples).astype(int)
    trestbps = np.clip(trestbps, 94, 200)
    
    # 5. Cholesterol (chol) (126 to 400)
    chol = np.random.normal(loc=240, scale=45, size=num_samples).astype(int)
    chol = np.clip(chol, 126, 400)
    
    # 6. Max Heart Rate (thalach) (71 to 202)
    thalach = np.random.normal(loc=150, scale=20, size=num_samples).astype(int)
    thalach = thalach - (age - 50) * 0.5
    thalach = np.clip(thalach, 71, 202).astype(int)
    
    # Calculate linear combination for logistic regression probability
    # intercept adjusted to balance the classes
    z = (
        -1.2
        + 0.04 * (age - 50)           # Higher age increases risk (+ relative to mean 50)
        + 0.8 * sex                   # Male increases risk
        + 0.6 * cp                    # Higher chest pain type indicator increases risk
        + 0.015 * (trestbps - 130)    # Elevated BP increases risk (+ relative to mean 130)
        + 0.006 * (chol - 220)        # Elevated cholesterol increases risk (+ relative to mean 220)
        - 0.035 * (thalach - 150)     # Lower max heart rate increases risk
    )
    
    # Sigmoid function for probability
    probs = 1 / (1 + np.exp(-z))
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.08, size=num_samples)
    target = (probs + noise >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'target': target
    })
    
    return df

def train_and_save_model():
    print("Generating balanced synthetic heart disease dataset...")
    df = generate_synthetic_data(num_samples=1500, random_state=42)
    
    X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Logistic Regression model with scaling pipeline...")
    # Create standardizing pipeline to handle scaling seamlessly during prediction
    model_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, random_state=42)
    )
    
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate model
    train_preds = model_pipeline.predict(X_train)
    test_preds = model_pipeline.predict(X_test)
    
    print(f"Train Accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, test_preds))
    
    # Check class balance
    print(f"Class distribution: {df['target'].value_counts(normalize=True).to_dict()}")
    
    # Save the pipeline to model.pkl
    print("Saving model pipeline to 'model.pkl'...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print("Model pipeline successfully saved!")

if __name__ == '__main__':
    train_and_save_model()
