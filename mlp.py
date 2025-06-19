import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

base_dir = 'output_lexical_metrics'            
mmse_path = 'mmse_input.csv'          

print(f"Checking data directory: {base_dir}")
if not os.path.isdir(base_dir):
    raise FileNotFoundError(f"Data directory '{base_dir}' not found.")
for subgroup in ['cc', 'cd']:
    path = os.path.join(base_dir, subgroup)
    print(f" - Subfolder '{path}': {'FOUND' if os.path.isdir(path) else 'MISSING'}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Expected '{path}' not found.")

print("\nLoading lexical metrics CSVs...")
filenames, data_rows, labels = [], [], []
for label in ['cc', 'cd']:
    folder = os.path.join(base_dir, label)
    files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    print(f" * {len(files)} in '{label}'")
    for fname in files:
        df = pd.read_csv(os.path.join(folder, fname))
        data_rows.append(df.iloc[0])
        labels.append(label)
        # strip extension to match mmse filenames
        base_name = os.path.splitext(fname)[0]
        filenames.append(base_name)

lex_df = pd.DataFrame(data_rows)
lex_df['label'] = labels
lex_df['filename'] = filenames
print(f"Total lexical samples: {len(lex_df)}")

#  Load and preprocess MMSE scores
print(f"\nLoading MMSE scores from: {mmse_path}")
mmse_df = pd.read_csv(mmse_path)
print("Original MMSE columns:", mmse_df.columns.tolist())

# Ensure 'filename' and 'mmse_score' columns exist
# and strip .csv if included
if 'filename' not in mmse_df.columns:
    raise ValueError("MMSE file must have a 'filename' column.")
mmse_df['filename'] = mmse_df['filename'].apply(lambda x: os.path.splitext(x)[0])
if 'mmse_score' not in mmse_df.columns:
    raise ValueError("MMSE file must have a 'mmse_score' column.")

print("MMSE data after processing filenames:")
print(mmse_df.head())

# 5. Merge lexical metrics with MMSE
merged = pd.merge(lex_df, mmse_df, on='filename', how='left')
print(f"\nAfter merge, data shape: {merged.shape}")
missing_mmse = merged['mmse_score'].isna().sum()
print(f"MMSE missing for {missing_mmse} samples")

# 6. Prepare feature matrix and target
X = merged.drop(columns=['label', 'filename'])
y = merged['label'].map({'cc': 0, 'cd': 1})

# 7. Impute missing values with mean
print("\nImputing missing values via column means...")
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("Missing after imputation:", X.isna().sum().sum())

# 8. Split, standardize, and train MLP
print("\nSplitting data (80% train / 20% test)…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Standardizing features…")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nInitializing and training MLPClassifier…")
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    verbose=True,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 9. Evaluate
print("\nEvaluating on test set…")
y_pred = mlp.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nFinal training loss: {mlp.loss_curve_[-1]:.4f}")
