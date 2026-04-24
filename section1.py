import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('Students_Performance_data_set.xlsx')

print("\n========== DATASET OVERVIEW ==========")
print(df.head())

# -----------------------------------------
# DATA CLEANING
# -----------------------------------------

print("\n========== DATA CLEANING ==========")
print("Cleaning: Handling missing values and duplicates.")

df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum())

# -----------------------------------------
# TARGET VARIABLE VISUALIZATION
# -----------------------------------------

print("\n========== TARGET VARIABLE ANALYSIS ==========")

plt.figure()
plt.hist(df['current_cgpa'], bins=30)
plt.title("CGPA Distribution")
plt.savefig("target_variable_histogram.png")
plt.show()

plt.figure()
plt.boxplot(df['current_cgpa'])
plt.title("CGPA Boxplot")
plt.savefig("target_variable_boxplot.png")
plt.show()

# -----------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------

print("\n========== FEATURE EXTRACTION ==========")

df['study_effectiveness'] = df['attendance'] * df['hrs_study'] / 100
df['academic_progress'] = df['current_cgpa'] - df['prev_sgpa']

print("New features created.")

# -----------------------------------------
# FEATURE SELECTION
# -----------------------------------------

print("\n========== FEATURE SELECTION ==========")

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation = df[numerical_cols].corr()['current_cgpa'].sort_values(ascending=False)

selected_features = list(correlation[(abs(correlation) > 0.1) &
                                     (correlation.index != 'current_cgpa')].index)

print("Selected Features:", selected_features)

# -----------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------

corr_matrix = df[numerical_cols].corr()

plt.figure()
plt.imshow(corr_matrix)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# -----------------------------------------
# DATA TRANSFORMATION
# -----------------------------------------

print("\n========== DATA TRANSFORMATION ==========")

df_encoded = pd.get_dummies(df, drop_first=True)

# -----------------------------------------
# DATA SCALING
# -----------------------------------------

print("\n========== DATA SCALING ==========")

X = df_encoded[selected_features]
y = df_encoded['current_cgpa']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n========== DATA SPLIT ==========")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# -----------------------------------------
# RBFN MODEL
# -----------------------------------------

print("\n========== MODEL TRAINING ==========")

from sklearn.cluster import KMeans

def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c)**2 / (2 * s**2))

class RBFN:
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        dists = []
        for i in range(len(self.centers)):
            for j in range(i+1, len(self.centers)):
                dists.append(np.linalg.norm(self.centers[i] - self.centers[j]))
        self.sigma = np.mean(dists)

        G = np.zeros((X.shape[0], self.k))
        for i in range(X.shape[0]):
            for j in range(self.k):
                G[i, j] = rbf(X[i], self.centers[j], self.sigma)

        self.weights = np.linalg.pinv(G).dot(y)

    def predict(self, X):
        G = np.zeros((X.shape[0], self.k))
        for i in range(X.shape[0]):
            for j in range(self.k):
                G[i, j] = rbf(X[i], self.centers[j], self.sigma)

        return G.dot(self.weights)

model = RBFN(k=15)
model.fit(X_train, y_train.values)

print("Model trained successfully.")

# -----------------------------------------
# TESTING
# -----------------------------------------

print("\n========== MODEL TESTING ==========")

y_pred = model.predict(X_test)

print("Sample Predictions:", y_pred[:5])

print("\n========== MODEL EVALUATION ==========")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

# -----------------------------------------
# MODEL EVALUATION GRAPH
# -----------------------------------------

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CGPA")
plt.ylabel("Predicted CGPA")
plt.title("Model Evaluation")
plt.savefig("model_evaluation.png")
plt.show()

# -----------------------------------------
# CROSS VALIDATION
# -----------------------------------------

print("\n========== CROSS VALIDATION ==========")

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
scores = []

for train_idx, test_idx in kf.split(X_scaled):
    model = RBFN(k=15)
    model.fit(X_scaled[train_idx], y.iloc[train_idx])
    pred = model.predict(X_scaled[test_idx])
    scores.append(r2_score(y.iloc[test_idx], pred))

print("Average 5-Fold R2 Score:", np.mean(scores))

# -----------------------------------------
# HYPERPARAMETER TUNING
# -----------------------------------------

print("\n========== HYPERPARAMETER TUNING ==========")

k_values = [5, 10, 15, 20, 25]
results = []

for k in k_values:
    model = RBFN(k=k)
    model.fit(X_train, y_train.values)
    pred = model.predict(X_test)
    results.append(r2_score(y_test, pred))

print("Tuning Results:", results)

plt.plot(k_values, results, marker='o')
plt.xlabel("Number of Centers (k)")
plt.ylabel("R2 Score")
plt.title("RBFN Hyperparameter Tuning")
plt.savefig("hyperparameter_tuning.png")
plt.show()