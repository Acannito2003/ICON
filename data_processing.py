import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def preprocess_heart_dataset(file_path):
    """
    Carica, pulisce e pre-elabora il dataset relativo alle malattie cardiache.
    """
    df = pd.read_csv(file_path).copy()
    df_cleaned = df.drop_duplicates().copy()
    df_cleaned.fillna(df_cleaned.mean(), inplace=True)
    df_cleaned['sex'] = df_cleaned['sex'].replace({0: 'Female', 1: 'Male'}).astype(str)
    thal_mapping = {1: 'Normale', 2: 'Difettoso fisso', 3: 'Difettoso reversibile'}
    df_cleaned['thal'] = df_cleaned['thal'].replace(thal_mapping).astype(str)
    df_encoded = pd.get_dummies(df_cleaned, columns=['sex', 'cp', 'restecg', 'slope', 'ca', 'thal'], drop_first=True)
    return df_encoded

def standardize_dataset(df):
    """
    Standardizza le feature continue del dataset.
    """
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features].astype(float))
    return df

def discretize_dataset(df):
    """
    Discretizza le feature continue.
    """
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
    continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df.loc[:, continuous_columns] = discretizer.fit_transform(df[continuous_columns])
    return df

def normalize_dataset(df, target):
    """
    Normalizza le feature continue nell'intervallo [0,1].
    """
    X = df.drop(columns=[target])
    y = df[target]
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns].astype('float64'))
    df_normalized = pd.DataFrame(X, columns=X.columns)
    df_normalized[target] = y
    return df_normalized

def over_sampling(df, target):
    """
    Applica SMOTE per l'oversampling al fine di bilanciare le classi.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(df.drop(columns=[target]), df[target])
    df_resampled = pd.DataFrame(X_resampled, columns=df.drop(columns=[target]).columns)
    df_resampled[target] = y_resampled
    return df_resampled

def prepare_dataset_for_unsupervised(df, target):
    """
    Prepara il dataset per l'apprendimento non supervisionato, normalizzando i dati
    e rimuovendo la feature target.
    """
    df = normalize_dataset(df, target)
    df = df[df[target] != 0]  # Mantiene solo i pazienti con la malattia
    df = df.drop(columns=[target])
    return df

def save_dataset(df, output_path):
    """
    Salva il dataset pre-processato in un file CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Dataset salvato in {output_path}")

"""
def main():
    # Percorso del dataset
    dataset_path = os.path.join('datasets', 'heart.csv')

    # Carica il dataset
    df = pd.read_csv(dataset_path)

    # Discretizza il dataset
    discretized_df = discretize_dataset(df)

    # Salva il dataset discretizzato nella cartella datasets
    discretized_dataset_path = os.path.join('datasets', 'heart_processed.csv')
    discretized_df.to_csv(discretized_dataset_path, index=False)
    print(f"Dataset discretizzato salvato come '{discretized_dataset_path}'")


if __name__ == "__main__":
    main()
    
"""
