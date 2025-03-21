import pickle
import time
import os
import networkx as nx
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BayesianEstimator, BicScore
from pgmpy.inference import VariableElimination
from pgmpy.metrics import correlation_score, log_likelihood_score
from pgmpy.models import BayesianNetwork
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from plot import visualize_bayesian_network


def create_bayesian_network(df):
    """Apprendimento della struttura della Rete Bayesiana per l'analisi delle malattie cardiache"""

    hc_k2 = HillClimbSearch(df)
    k2_model = hc_k2.estimate(scoring_method='k2score', max_iter=75)  # Prova con 75 iterazioni


    model = BayesianNetwork(k2_model.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)

    # Visualizzazione della rete bayesiana
    visualize_bayesian_network(model)


    return model


def query_report(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    """Effettua inferenza sulla rete bayesiana basata su evidenze per l'analisi delle malattie cardiache"""

    print(f"Query: {desc}")
    print("Risposta:")
    print(infer.query(variables=variables,
                      evidence=evidence,
                      elimination_order=elimination_order,
                      show_progress=show_progress))


def generate_random_example(bayesianNetwork: BayesianNetwork):
    """Genera un esempio randomico di un paziente con malattia cardiaca sulla base della rete bayesiana"""
    return bayesianNetwork.simulate(n_samples=1)


def predict(bayesianNetwork: BayesianNetwork, example, target):
    """Effettua la predizione del valore di una variabile target sulla base della rete bayesiana"""

    # Assicuriamoci che i valori nel dizionario siano numerici (interi)
    example = {key: int(value) for key, value in example.items()}

    inference = VariableElimination(bayesianNetwork)

    # Verifica che il target esista nel modello
    if target not in bayesianNetwork.nodes():
        print(f"Errore: La variabile '{target}' non esiste nella rete bayesiana.")
        return

    # Stampa i dati usati per la predizione
    print(f"\nEvidenze utilizzate per l'inferenza: {example}")

    # Eseguire la query di inferenza
    result = inference.query(variables=[target], evidence=example, elimination_order='MinFill')

    print("\nRisultato dell'inferenza:")
    print(result)


def visualize_info(bayesianNetwork: BayesianNetwork):
    """Stampa le distribuzioni di probabilità condizionate (CPD) per ogni variabile nella rete"""

    for cpd in bayesianNetwork.get_cpds():
        print(f'CPD di {cpd.variable}:')
        print(cpd, '\n')


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


# Modifica la funzione main per caricare il dataset
def main():
    # Carica il dataset dal file che hai fornito
    dataset_path = 'datasets/heart_discretized.csv'  # Percorso del file caricato
    df = pd.read_csv(dataset_path)

    # Menu delle scelte
    while True:
        print("\nSeleziona una delle seguenti opzioni:")
        print("1. Crea e visualizza la rete bayesiana")
        print("2. Genera un esempio casuale di paziente")
        print("3. Effettua una previsione per una variabile target")
        print("4. Esegui una query sulla rete bayesiana")
        print("5. Esci")

        scelta = input("Inserisci il numero della tua scelta (1, 2, 3, 4, 5): ")

        if scelta == "1":
            print("Creando la rete bayesiana...")
            model = create_bayesian_network(df)

        elif scelta == "2":
            print("Generando un esempio casuale di paziente...")
            model = create_bayesian_network(df)  # Ricrea la rete se non è già creata
            random_example = generate_random_example(model)
            print(f"Esempio casuale generato: {random_example.to_dict()}")






        elif scelta == "3":

            model = create_bayesian_network(df)  # Creiamo la rete bayesiana

            # Genera un esempio randomico discretizzato

            raw_example = generate_random_example(model).to_dict()

            # Convertiamo l'esempio in un dizionario semplice con valori numerici

            random_example = {}

            for key, value in raw_example.items():

                if isinstance(value, dict):  # Se è un dizionario, estrai il primo valore

                    random_example[key] = int(next(iter(value.values())))

                elif isinstance(value, list):  # Se è una lista, prendi il primo valore

                    random_example[key] = int(value[0])

                else:

                    random_example[key] = int(value)  # Se è già un numero, lo converte direttamente

            # Variabile target fissa: trestbps

            missing_feature = "trestbps"

            if missing_feature not in df.columns:
                continue

            # Rimuove la variabile target dall'esempio per testare l'inferenza

            if missing_feature in random_example:

                del random_example[missing_feature]

                # Eseguiamo l'inferenza sulla variabile mancante

                inference = VariableElimination(model)

                result = inference.query(variables=[missing_feature], evidence=random_example,
                                         elimination_order='MinFill')

                # Stampa SOLO il risultato della predizione

                for label, prob in zip(result.state_names[missing_feature], result.values):
                    print(f"trestbps({label}) → Probabilità: {prob:.4f}")





        elif scelta == "4":

            model = create_bayesian_network(df)  # Creiamo la rete bayesiana

            inference = VariableElimination(model)

            # Variabili target da interrogare

            target_variable = "thalach"

            # Definizione delle evidenze

            evidence = {

                "age": 3,

                "chol": 2,

                "cp": 3,

                "slope": 0

            }

            # Stampa delle evidenze usate nella query

            print(f"\nEvidenze utilizzate per l'inferenza: {evidence}")

            # Esegue la query sulla rete bayesiana

            result = inference.query(variables=[target_variable], evidence=evidence, elimination_order="MinFill")

            # Stampa SOLO il risultato della predizione

            print("\nRisultati dell'interrogazione sulla variabile 'thalach':")

            for label, prob in zip(result.state_names[target_variable], result.values):
                print(f"thalach({label}) → Probabilità: {prob:.6f}")

        elif scelta == "5":
            print("Uscita dal programma.")
            break

        else:
            print("Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()