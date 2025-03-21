import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from plot import *





def return_best_hyperparameters_tree_based(df, target_feature):
    """Ricerca dei migliori iperparametri per i modelli ad albero tramite GridSearch e Cross Validation."""
    X = df.drop(target_feature, axis=1).to_numpy()
    y = df[target_feature].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    hyperparameters = {
        'DecisionTree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [7, 10, 12],
            'min_samples_split': [8, 10, 15],
            'min_samples_leaf': [5, 7, 10],
        },
        'RandomForest': {
            'criterion': ['gini', 'entropy'],
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [3, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 10],
            'min_samples_split': [5, 8, 12],
            'min_samples_leaf': [3, 5, 7, 10]
        }
    }

    best_params = {}
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, hyperparameters[model_name], cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_

    return best_params


def train_model_kfold_tree_based(df, target_feature):
    """Addestramento dei modelli supervisionati basati su alberi."""
    best_params = return_best_hyperparameters_tree_based(df, target_feature)
    X = df.drop(target_feature, axis=1).to_numpy()
    y = df[target_feature].to_numpy()
    cv = RepeatedKFold(n_splits=5, n_repeats=5)
    models = {
        'DecisionTree': DecisionTreeClassifier(**best_params['DecisionTree']),
        'RandomForest': RandomForestClassifier(**best_params['RandomForest']),
        'GradientBoosting': GradientBoostingClassifier(**best_params['GradientBoosting'])
    }

    results = {}
    for model_name, model in models.items():
        scores = {
            'accuracy': cross_val_score(model, X, y, scoring='accuracy', cv=cv).mean(),
            'precision': cross_val_score(model, X, y, scoring='precision_macro', cv=cv).mean(),
            'recall': cross_val_score(model, X, y, scoring='recall_macro', cv=cv).mean(),
            'f1': cross_val_score(model, X, y, scoring='f1_macro', cv=cv).mean()
        }
        results[model_name] = scores

    return results


def return_best_hyperparameters_lg(df, target_feature):
    """Ricerca dei migliori iperparametri per la regressione logistica."""
    X = df.drop(target_feature, axis=1).to_numpy()
    y = df[target_feature].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    hyperparameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100000, 150000]
    }

    grid_search = GridSearchCV(LogisticRegression(), hyperparameters, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


def train_model_kfold_lg(df, target_feature):
    """Addestramento della regressione logistica."""
    best_params = return_best_hyperparameters_lg(df, target_feature)
    X = df.drop(target_feature, axis=1).to_numpy()
    y = df[target_feature].to_numpy()
    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    model = LogisticRegression(**best_params)
    scores = {
        'accuracy': cross_val_score(model, X, y, scoring='accuracy', cv=cv).mean(),
        'precision': cross_val_score(model, X, y, scoring='precision_macro', cv=cv).mean(),
        'recall': cross_val_score(model, X, y, scoring='recall_macro', cv=cv).mean(),
        'f1': cross_val_score(model, X, y, scoring='f1_macro', cv=cv).mean()
    }

    return scores


def error_variance_and_std(models, X_train, X_test, y_train, y_test):
    """
    Funzione per calcolare la varianza degli errori e la deviazione standard degli errori
    per ciascun modello.

    models: dizionario contenente i modelli da valutare
    X_train: Dati di addestramento (caratteristiche)
    X_test: Dati di test (caratteristiche)
    y_train: Dati di addestramento (target)
    y_test: Dati di test (target)

    Ritorna un dizionario contenente la varianza e la deviazione standard per ogni modello.
    """
    error_stats = {}

    for model_name, model in models.items():
        # Allenare il modello
        model.fit(X_train, y_train)

        # Previsione sui dati di test
        y_pred = model.predict(X_test)

        # Calcolare gli errori (differenza tra il vero valore e la previsione)
        errors = y_test - y_pred

        # Calcolare la varianza degli errori
        error_variance = np.var(errors)

        # Calcolare la deviazione standard degli errori
        error_std_dev = np.std(errors)

        # Memorizzare i risultati nel dizionario
        error_stats[model_name] = {
            'error_variance': error_variance,
            'error_std_dev': error_std_dev
        }

        # Stampare i risultati
        print(f"Modello: {model_name}")
        print(f"  Varianza degli errori: {error_variance:.4f}")
        print(f"  Deviazione standard degli errori: {error_std_dev:.4f}")

    return error_stats


if __name__ == "__main__":
    while True:
        # Menu per scegliere quale funzione testare
        print("\nScegli una funzione da testare:")
        print("1. Trova i migliori iperparametri per modelli ad albero")
        print("2. Addestra i modelli ad albero con K-fold")
        print("3. Trova i migliori iperparametri per la regressione logistica")
        print("4. Addestra la regressione logistica con K-fold")
        print("5. Crea tutte le curve di apprendimento per Decision Tree, Random Forest e Gradient Boosting")
        print("6. Crea la curva di apprendimento per la logistica")
        print("7. Crea il grafico delle metriche per Decision Tree, Random Forest e Gradient Boosting")
        print("8. Crea il grafico delle metriche per la logistica")
        print("9 Trova varianza e deviazione standard degli errori per modelli ad albero")
        print("10 Trova varianza e deviazione standard degli errori per la regressione logistica")
        print("11 Crea grafico della distribuzione della variabile target con e senza Oversampling")
        print ("12. Esci dal programma")
        scelta = input("Inserisci il numero della funzione che vuoi testare: ")

        # Esegui la funzione scelta
        if scelta == "1":
            # Trova i migliori iperparametri per modelli ad albero
            print("Ricerca dei migliori iperparametri per modelli ad albero...")
            df = pd.read_csv('datasets/heart_processed.csv')  # Percorso del dataset nella cartella datasets
            target_feature = 'target'  # Sostituisci con il nome della colonna target
            best_params_tree = return_best_hyperparameters_tree_based(df, target_feature)
            print("Migliori iperparametri per i modelli ad albero:", best_params_tree)

        elif scelta == "2":
            # Addestra i modelli ad albero con K-fold
            print("\nAddestramento dei modelli ad albero con K-fold...")
            df = pd.read_csv('datasets/heart_processed.csv')  # Percorso del dataset nella cartella datasets
            target_feature = 'target'  # Sostituisci con il nome della colonna target
            tree_results = train_model_kfold_tree_based(df, target_feature)
            print("Risultati dei modelli ad albero:", tree_results)

        elif scelta == "3":
            # Trova i migliori iperparametri per la regressione logistica
            print("\nRicerca dei migliori iperparametri per la regressione logistica...")
            df = pd.read_csv('datasets/heart_normalized.csv')  # Percorso del dataset nella cartella datasets
            target_feature = 'target'  # Sostituisci con il nome della colonna target
            best_params_lg = return_best_hyperparameters_lg(df, target_feature)
            print("Migliori iperparametri per la regressione logistica:", best_params_lg)

        elif scelta == "4":
            # Addestra la regressione logistica con K-fold
            print("\nAddestramento della regressione logistica con K-fold...")
            df = pd.read_csv('datasets/heart_normalized.csv')  # Percorso del dataset nella cartella datasets
            target_feature = 'target'  # Sostituisci con il nome della colonna target
            lg_results = train_model_kfold_lg(df, target_feature)
            print("Risultati della regressione logistica:", lg_results)

        elif scelta == "5":
            # Crea tutte le curve di apprendimento per vari modelli
            print("\nCreazione delle curve di apprendimento per Decision Tree, Random Forest e Gradient Boosting")
            df = pd.read_csv('datasets/heart_processed.csv')  # Carica il dataset
            target_feature = 'target'  # Nome della colonna target

            # Separare X (caratteristiche) e y (target)
            X = df.drop(columns=[target_feature])  # Features
            y = df[target_feature]  # Target

            # Definire i modelli
            models = {
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            }

            # Genera le curve di apprendimento per tutti i modelli
            plot_all_learning_curves(models, X, y, metodo="Metodo")

        elif scelta == "6":
            # Crea tutte la curva di apprendimento di logistic
            print("\nCreazione della curva di apprendimento per la logistic")
            df = pd.read_csv('datasets/heart_normalized.csv')  # Carica il dataset
            target_feature = 'target'  # Nome della colonna target

            # Separare X (caratteristiche) e y (target)
            X = df.drop(columns=[target_feature])  # Features
            y = df[target_feature]  # Target

            # Definire i modelli
            models = {
                'Logistic Regression': LogisticRegression(),
            }

            # Genera le curve di apprendimento per tutti i modelli
            plot_all_learning_curves(models, X, y, metodo="Metodo")


        elif scelta == "7":

            # Crea grafico delle metriche per tutti i modelli

            print("\nCreazione del grafico delle metriche per Decision Tree, Random Forest e Gradient Boosting")

            df = pd.read_csv('datasets/heart_processed.csv')  # Carica il dataset

            target_feature = 'target'

            # Separare X (caratteristiche) e y (target)

            X = df.drop(columns=[target_feature])  # Features

            y = df[target_feature]  # Target

            # Suddividere il dataset in set di addestramento e test

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Definire i modelli

            models = {


                'Decision Tree': DecisionTreeClassifier(),

                'Random Forest': RandomForestClassifier(),

                'Gradient Boosting': GradientBoostingClassifier()

            }

            # Dizionario per memorizzare le metriche dei modelli

            models_metrics = {}

            # Calcolare le metriche per ogni modello

            for model_name, model in models.items():
                # Allenare il modello

                model.fit(X_train, y_train)

                # Previsione sui dati di test

                y_pred = model.predict(X_test)

                # Calcolare le metriche

                accuracy = accuracy_score(y_test, y_pred)

                precision = precision_score(y_test, y_pred)

                recall = recall_score(y_test, y_pred)

                f1 = f1_score(y_test, y_pred)

                # Memorizzare i risultati nel dizionario

                models_metrics[model_name] = {

                    'accuracy': accuracy,

                    'precision': precision,

                    'recall': recall,

                    'f1_score': f1

                }

            # Stampa le metriche per tutti i modelli

            plot_all_model_metrics(models_metrics)
        elif scelta == "8":

            # Crea grafico delle metriche per tutti i modelli

            print("\nCreazione del grafico delle metriche per la logistic")

            df = pd.read_csv('datasets/heart_normalized.csv')  # Carica il dataset

            target_feature = 'target'

            # Separare X (caratteristiche) e y (target)

            X = df.drop(columns=[target_feature])  # Features

            y = df[target_feature]  # Target

            # Suddividere il dataset in set di addestramento e test

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Definire i modelli

            models = {


                'Logistic Regression': LogisticRegression(),



            }

            # Dizionario per memorizzare le metriche dei modelli

            models_metrics = {}

            # Calcolare le metriche per ogni modello

            for model_name, model in models.items():
                # Allenare il modello

                model.fit(X_train, y_train)

                # Previsione sui dati di test

                y_pred = model.predict(X_test)

                # Calcolare le metriche

                accuracy = accuracy_score(y_test, y_pred)

                precision = precision_score(y_test, y_pred)

                recall = recall_score(y_test, y_pred)

                f1 = f1_score(y_test, y_pred)

                # Memorizzare i risultati nel dizionario

                models_metrics[model_name] = {

                    'accuracy': accuracy,

                    'precision': precision,

                    'recall': recall,

                    'f1_score': f1

                }

            # Stampa le metriche per tutti i modelli

            plot_all_model_metrics(models_metrics)

        elif scelta == "9":
            # Carica il dataset
            df = pd.read_csv('datasets/heart_processed.csv')
            target_feature = 'target'

            # Separare X (caratteristiche) e y (target)
            X = df.drop(columns=[target_feature])
            y = df[target_feature]

            # Suddividere il dataset in set di addestramento e test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Definire i modelli
            models = {
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            }

            # Calcolare la varianza e la deviazione standard degli errori per tutti i modelli
            error_stats = error_variance_and_std(models, X_train, X_test, y_train, y_test)

        elif scelta == "10":
            # Carica il dataset
            df = pd.read_csv('datasets/heart_normalized.csv')
            target_feature = 'target'

            # Separare X (caratteristiche) e y (target)
            X = df.drop(columns=[target_feature])
            y = df[target_feature]

            # Suddividere il dataset in set di addestramento e test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Definire i modelli
            models = {
                'Logistic Regression': LogisticRegression(),
            }

            # Calcolare la varianza e la deviazione standard degli errori per tutti i modelli
            error_stats = error_variance_and_std(models, X_train, X_test, y_train, y_test)

        elif scelta == "11":

            # Visualizza la distribuzione della variabile target
            print("\nVisualizzazione della distribuzione della variabile target senza e con oversampling...")

            # Carica il dataset
            df = pd.read_csv('datasets/heart_processed.csv')  # Percorso del dataset nella cartella datasets
            target_feature = 'target'  # Sostituisci con il nome della colonna target
            metodo = "Metodo"  # O qualsiasi descrizione tu voglia

            # Visualizza il grafico senza oversampling
            visualize_target_distribution(df, target_feature, metodo)

            # Separare X (caratteristiche) e y (target)
            X = df.drop(columns=[target_feature])  # Features
            y = df[target_feature]  # Target

            # Applicare l'oversampling (SMOTE) sui dati di addestramento
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Creare un nuovo DataFrame con i dati resampled per visualizzare la distribuzione
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled[target_feature] = y_resampled

            # Visualizza il grafico con oversampling
            visualize_target_distribution(df_resampled, target_feature, metodo + "_oversampled")

        elif scelta == "12":
            # Esci dal programma
            print("Esci dal programma.")
            break


        else:
            print("Scelta non valida. Riprova.")