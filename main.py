import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
 
# 1. Load Dataset
df = pd.read_csv('Titanic-Dataset.csv')

# 2. Feature engineering
def feature_engineer(df):
    df = df.copy()

    # Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr',
                                       'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle','Miss')
    df['Title'] = df['Title'].replace('Ms','Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Cabin deck
    df['Deck'] = df['Cabin'].str[0].fillna('U')

    # Fill missing
    df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    return df

# 3. Define features
target = 'Survived'
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone','Deck']

updated_df = feature_engineer(df)
X = updated_df[features]
y = updated_df[target]


# 4. Preprocessing: scale numeric, one-hot categorical
numeric_features = ['Age','SibSp','Parch','Fare','FamilySize','IsAlone']
categorical_features = ['Pclass','Sex','Embarked','Title','Deck']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Pipeline: preprocessing + All Classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN':                 KNeighborsClassifier(),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(random_state=42),
    'SVM':                 SVC(random_state=42)
}
results = {}

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for name, clf in classifiers.items():
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc


print("\nAll Classifier Results:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")


# Define models and their param grids
# Note: pipeline params use clf__ prefix to target the classifier step
param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'clf__C':       [0.01, 0.1, 1, 10, 100],
            'clf__solver':  ['lbfgs', 'liblinear']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'clf__n_neighbors': [3, 5, 7, 9, 11],
            'clf__weights':     ['uniform', 'distance'],
            'clf__metric':      ['euclidean', 'manhattan']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'clf__max_depth':        [3, 5, 7, 10, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__criterion':        ['gini', 'entropy']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'clf__n_estimators':      [100, 200, 300],
            'clf__max_depth':         [3, 5, 7, None],
            'clf__min_samples_split': [2, 5, 10]
        }
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'clf__C':      [0.1, 1, 10, 100],
            'clf__kernel': ['rbf', 'linear'],
            'clf__gamma':  ['scale', 'auto']
        }
    }
}

# Loop through all models
tuned_results = {}

for name, config in param_grids.items():

    # Build pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', config['model'])
    ])

    # GridSearchCV with 5-fold cross validation
    grid_search = GridSearchCV(
        pipe,
        config['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,        # use all CPU cores, runs faster
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    tuned_results[name] = {
        'accuracy':    acc,
        'best_params': grid_search.best_params_,
        'model':       best_model
    }

    print(f"\n{'='*45}")
    print(f"Model: {name}")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Tuned Accuracy: {acc:.4f}")

# Final comparison — before vs after tuning
print(f"\n{'='*45}")
print(f"{'Model':<22} {'Before':>8} {'After':>8} {'Change':>8}")
print(f"{'='*45}")


for name, res in tuned_results.items():
    diff = res['accuracy'] - results[name]
    sign = '+' if diff > 0 else ''
    print(f"{name:<22} {results[name]:>8.4f} {res['accuracy']:>8.4f} {sign}{diff:>7.4f}")