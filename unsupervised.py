import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


# Impostiamo il numero massimo di thread a 1 per evitare conflitti nelle operazioni parallele
os.environ["OMP_NUM_THREADS"] = "1"


def prepare_dataset_for_unsupervised():
    """
    Carica il dataset 'heart_normalized' dalla cartella 'datasets', filtra i pazienti con la malattia,
    e rimuove la colonna target.
    """
    # Percorso del dataset
    dataset_path = os.path.join('datasets', 'heart_normalized.csv')  # Aggiungi l'estensione .csv se necessario
    df = pd.read_csv(dataset_path)

    target = 'target'  # Nome della colonna target, modifica se il nome è diverso

    # Filtra i pazienti con la malattia (escludendo i pazienti con target = 0)
    df = df[df[target] != 0]  # Mantiene solo i pazienti con la malattia

    # Rimuoviamo la colonna target
    df = df.drop(columns=[target])

    # Restituiamo il dataset pronto per il clustering come numpy array
    return df.values


def elbow_rule(dataSet):
    """Effettua la regola del gomito per trovare il numero ottimale k di cluster"""

    inertia = []
    maxK = 10

    for i in range(1, maxK):
        kmeans = KMeans(n_clusters=i, n_init=5, init='random')
        kmeans.fit(dataSet)
        inertia.append(kmeans.inertia_)

    # tramite KneeLocator troviamo il k ottimale
    kl = KneeLocator(range(1, maxK), inertia, curve="convex", direction="decreasing")

    # grafico della curva del gomito
    plt.plot(range(1, maxK), inertia, 'bx-')
    plt.scatter(kl.elbow, inertia[kl.elbow - 1], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Metodo del gomito per trovare il k ottimale')
    plt.legend()
    plt.savefig(f'plots/elbow_rule.png', dpi=300, bbox_inches='tight')
    plt.show()

    return kl.elbow


def clustering_kmeans(dataSet):
    """Effettua il clustering tramite l'algoritmo KMeans"""

    k = elbow_rule(dataSet)
    km = KMeans(n_clusters=k, n_init=10, init='random')
    km = km.fit(dataSet)

    validation(km, dataSet)

    etichette = km.labels_
    centroidi = km.cluster_centers_

    return etichette, centroidi


def validation(k_means, dataset):
    """Stampa le metriche di valutazione del clustering effettuato"""

    wcss = k_means.inertia_
    print("\nWCSS:", wcss)

    silhouette_avg = silhouette_score(dataset, k_means.labels_)
    print("Silhouette Score:", silhouette_avg)


def visualizza_cluster(dataSet, etichette, metodo):
    """Funzione per la visualizzazione dei cluster con PCA o t-SNE"""

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


def main():
    # Carica il dataset normalizzato
    dataset_path = os.path.join('datasets', 'heart_normalized.csv')
    df = pd.read_csv(dataset_path)

    target = 'target'  # Nome della colonna target

    # Prepara il dataset per l'apprendimento non supervisionato
    dataSet = prepare_dataset_for_unsupervised()


    print("Eseguendo clustering KMeans...")
    etichette, centroidi = clustering_kmeans(dataSet)

         # Visualizza i cluster utilizzando PCA e t-SNE
    visualizza_cluster(dataSet, etichette, "pca")  # Visualizzazione con PCA
    visualizza_cluster(dataSet, etichette, "tsne")  # Visualizzazione con t-SNE



if __name__ == "__main__":
    main()
