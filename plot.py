from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt


# Funzione per stampare le curve di apprendimento per vari modelli
def plot_all_learning_curves(models, X, y, metodo="Metodo"):
    """
    Crea e salva le curve di apprendimento per una lista di modelli.

    models: dizionario con i modelli (nome: modello)
    X: caratteristiche (dataframe)
    y: target (serie o array)
    metodo: metodo utilizzato (facoltativo)
    """
    for model_name, model in models.items():
        print(f"Creazione della curva di apprendimento per {model_name}...")
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy')

        train_errors = 1 - train_scores
        test_errors = 1 - test_scores

        mean_train_errors = 1 - np.mean(train_scores, axis=1)
        mean_test_errors = 1 - np.mean(test_scores, axis=1)

        # Grafico della curva di apprendimento
        plt.figure(figsize=(12, 6))
        plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='blue')
        plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
        plt.title(f'Curva di apprendimento per {model_name} ({metodo})')
        plt.xlabel('Dimensione del training set')
        plt.ylabel('Errore')
        plt.legend()
        plt.grid()

        # Salvataggio della curva nel file
        plt.savefig(f'plots/curva_{model_name}_{metodo}.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_model_metrics(title, model, metodo="Originale"):
    """
    Crea un grafico a barre per visualizzare le metriche del modello.

    title: nome del modello
    model: dizionario contenente le metriche ('accuracy', 'precision', 'recall', 'f1_score')
    metodo: descrizione del metodo
    """
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Estrarre i valori numerici dai dizionari (assumiamo che 'model' contenga un dizionario per le metriche)
    metric_vals = [model['accuracy'], model['precision'], model['recall'], model['f1_score']]

    # Verifica che metric_vals contenga valori numerici
    print(f"Metric Values for {title}: {metric_vals}")

    # Creazione del grafico a barre
    plt.figure(figsize=(8, 6))
    plt.bar(metric_labels, metric_vals, color=['blue', 'green', 'red', 'purple'])

    # Aggiungi titolo e etichette
    plt.title(f'Metriche del modello: {title} ({metodo})')
    plt.xlabel('Metriche')
    plt.ylabel('Valore')

    # Mostra il grafico
    plt.show()


def plot_all_model_metrics(models_metrics):
    """
    Funzione che stampa e visualizza le metriche per tutti i modelli.

    models_metrics: dizionario contenente i risultati delle metriche per ciascun modello.
    """
    for model_name, metrics in models_metrics.items():
        plot_model_metrics(model_name, metrics, metodo="Metodo")


# Funzione per la visualizzazione dei cluster (PCA o t-SNE)
def visualizza_cluster(dataSet, etichette, metodo):
    if metodo == "pca":
        reducer = PCA(n_components=2)
    elif metodo == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Metodo non valido. Usa 'pca' o 'tsne'.")

    dati_ridotti = reducer.fit_transform(dataSet)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(dati_ridotti[:, 0], dati_ridotti[:, 1], c=etichette, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Visualizzazione dei Cluster con {metodo.upper()}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")

    plt.savefig(f'plots/cluster_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Funzione per visualizzare la Rete Bayesiana
def visualize_bayesian_network(bayesianNetwork: BayesianNetwork):
    G = nx.MultiDiGraph(bayesianNetwork.edges())
    pos = nx.spring_layout(G, iterations=100, k=2, threshold=5, pos=nx.spiral_layout(G))

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=8, edge_color="blue", connectionstyle="arc3,rad=0.2")

    plt.title("Grafico della Rete Bayesiana")
    plt.savefig(f'plots/bayesian_network.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

# Funzione per visualizzare le curve di addestramento della Rete Neurale
def plot_curves_nn(history, best_params, metodo):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, best_params['epochs'] + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Train Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()

    plt.savefig(f'plots/curva_nn_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Funzione per visualizzare la distribuzione della feature target
def visualize_target_distribution(df, target, metodo):
    counts = df[target].value_counts()
    labels = counts.index.tolist()
    colors = plt.cm.Paired.colors

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribuzione della variabile target ({target})")

    plt.savefig(f'plots/distribuzione_{target}_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show()



