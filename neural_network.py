import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from itertools import product


# Definizione del modello
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, units_1, hidden_units_1, hidden_units_2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, units_1)  # Primo layer
        self.fc2 = nn.Linear(units_1, hidden_units_1)  # Secondo layer
        self.fc3 = nn.Linear(hidden_units_1, hidden_units_2)  # Terzo layer
        self.fc4 = nn.Linear(hidden_units_2, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Funzione di attivazione per output

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # attivazione ReLU
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # attivazione Sigmoid
        return x


# Funzione di addestramento
def train_neural_network(df, target_feature, units_1, hidden_units_1, hidden_units_2, optimizer_type, learning_rate,
                         batch_size, epochs):
    """Addestramento rete neurale"""



    # Converte tutte le colonne in numerico, sostituendo eventuali errori con NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Rimuove eventuali righe con NaN (dati mancanti o non numerici)
    df = df.dropna()



    # Controlla se il DataFrame è vuoto dopo la pulizia
    if df.empty:
        raise ValueError("❌ Errore: Il dataset è vuoto dopo la conversione. Verifica il contenuto del file CSV.")

    # Divisione dei dati in training e test
    train_data, test_data, train_targets, test_targets = train_test_split(
        df.drop(columns=[target_feature]), df[target_feature], test_size=0.3, random_state=42
    )

    # Converto i dati in tensori PyTorch
    train_data_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets.values, dtype=torch.float32).view(-1, 1)
    test_targets_tensor = torch.tensor(test_targets.values, dtype=torch.float32).view(-1, 1)

    # Impostazioni modello
    input_dim = train_data.shape[1]
    model = NeuralNetwork(input_dim=input_dim, units_1=units_1, hidden_units_1=hidden_units_1,
                          hidden_units_2=hidden_units_2)

    # Ottimizzatore
    optimizer = {
        'adam': optim.Adam(model.parameters(), lr=learning_rate),
        'sgd': optim.SGD(model.parameters(), lr=learning_rate),
        'rmsprop': optim.RMSprop(model.parameters(), lr=learning_rate)
    }[optimizer_type]

    # Funzione di perdita
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

    for epoch in range(epochs):
        model.train()
        outputs = model(train_data_tensor)
        loss = criterion(outputs, train_targets_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_preds = (outputs > 0.5).float()
        train_acc = (train_preds == train_targets_tensor).float().mean()

        train_loss_list.append(loss.item())
        train_acc_list.append(train_acc.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(test_data_tensor)
            val_loss = criterion(val_outputs, test_targets_tensor)
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == test_targets_tensor).float().mean()

        val_loss_list.append(val_loss.item())
        val_acc_list.append(val_acc.item())

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc.item():.4f}, Val Accuracy: {val_acc.item():.4f}')

    history = {'loss': train_loss_list, 'val_loss': val_loss_list, 'accuracy': train_acc_list,
               'val_accuracy': val_acc_list}
    return model, history


# Funzione per generare il classification report
def generate_classification_report(model, val_data_tensor, val_targets_tensor, test_data_tensor, test_targets_tensor):
    """
    Genera un report di classificazione per il validation set e il test set, includendo Accuracy.
    """
    model.eval()
    with torch.no_grad():
        # Predizioni sul validation set
        val_outputs = model(val_data_tensor)
        val_predictions = (val_outputs > 0.5).int()

        # Predizioni sul test set
        test_outputs = model(test_data_tensor)
        test_predictions = (test_outputs > 0.5).int()

    # Converto i tensori in array numpy per sklearn
    y_val_true = val_targets_tensor.numpy().flatten()
    y_val_pred = val_predictions.numpy().flatten()
    y_test_true = test_targets_tensor.numpy().flatten()
    y_test_pred = test_predictions.numpy().flatten()

    # Calcolo i classification report
    val_report = classification_report(y_val_true, y_val_pred, target_names=["Diagnosi Negativa", "Diagnosi Positiva"],
                                       output_dict=True)
    test_report = classification_report(y_test_true, y_test_pred,
                                        target_names=["Diagnosi Negativa", "Diagnosi Positiva"], output_dict=True)

    # Calcolo l'Accuracy
    val_accuracy = accuracy_score(y_val_true, y_val_pred)
    test_accuracy = accuracy_score(y_test_true, y_test_pred)

    print("\n📊 Report di classificazione per il Validation Set:\n")
    print_classification_report(val_report, val_accuracy)

    print("\n📊 Report di classificazione per il Test Set:\n")
    print_classification_report(test_report, test_accuracy)

    return val_report, test_report, val_accuracy, test_accuracy


def print_classification_report(report, accuracy):
    """ Stampa il classification report in modo formattato, includendo l'Accuracy """
    print(f"{'':<20} Precision    Recall    F1 Score    Support")
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Filtra solo i valori numerici
            print(
                f"{label:<20} {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}      {metrics['support']}")
    print(f"\n{'Accuracy':<20} {accuracy:.2f}")

def evaluate_model(model, test_data_tensor, test_targets_tensor, dataset_name="Test Set"):
    """Calcola Precision, Recall e F1-score"""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data_tensor)
        predictions = (outputs > 0.5).int()  # Converte le probabilità in 0/1

    # Converto i tensori in numpy array per sklearn
    y_true = test_targets_tensor.numpy().flatten()
    y_pred = predictions.numpy().flatten()

    # Calcolo il classification report
    report = classification_report(y_true, y_pred, target_names=["Diagnosi Negativa", "Diagnosi Positiva"], output_dict=True)

    # Stampiamo i risultati in formato tabellare
    print(f"\n📊 Report di classificazione per il {dataset_name}:\n")
    print(f"{'':<20} Precision    Recall    F1 Score    Support")
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Filtra solo i valori numerici
            print(f"{label:<20} {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}      {metrics['support']}")

    return report


def grid_search(df, target_feature, param_grid):
    """Trova i migliori iperparametri usando Grid Search"""

    best_score = 0
    best_params = None
    best_model = None

    # Genera tutte le combinazioni possibili di iperparametri
    param_combinations = list(product(
        param_grid['hidden_units_1'],
        param_grid['hidden_units_2'],
        param_grid['optimizer'],
        param_grid['learning_rate']
    ))

    print(f"\n🔍 Testando {len(param_combinations)} combinazioni di iperparametri...\n")

    for hidden_units_1, hidden_units_2, optimizer_type, learning_rate in param_combinations:
        print(f"\n▶️ Testando: hidden_units_1={hidden_units_1}, hidden_units_2={hidden_units_2}, optimizer={optimizer_type}, lr={learning_rate}")

        # Addestriamo il modello con gli iperparametri attuali
        model, history = train_neural_network(df, target_feature, 16, hidden_units_1, hidden_units_2, optimizer_type, learning_rate, 32, 50)

        # Otteniamo la miglior accuratezza sulla validation
        max_val_acc = max(history['val_accuracy'])

        # Se l'accuratezza è la migliore trovata finora, aggiorniamo i migliori iperparametri
        if max_val_acc > best_score:
            best_score = max_val_acc
            best_params = {
                'hidden_units_1': hidden_units_1,
                'hidden_units_2': hidden_units_2,
                'optimizer': optimizer_type,
                'learning_rate': learning_rate
            }
            best_model = model

    print("\n✅ Migliori iperparametri trovati:")
    print(best_params)
    print(f"🏆 Miglior Val Accuracy: {best_score:.4f}")

    return best_model, best_params

# Funzione per visualizzare le curve di addestramento
def plot_curves_nn(history, best_params, metodo):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, best_params['epochs'] + 1), history['loss'], label='Train Loss', color='blue')
    plt.plot(range(1, best_params['epochs'] + 1), history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, best_params['epochs'] + 1), history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(range(1, best_params['epochs'] + 1), history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()

    plt.savefig(f'plots/curva_nn_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show()


# Funzione per preparare i tensori con una suddivisione coerente

def prepare_tensors(df, target_feature):
    # Suddivisione stratificata in training (70%) e test (30%)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_feature])

    # Suddivisione ulteriore per ottenere un validation set (20% del training set)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df[target_feature])

    # Separazione feature-target
    train_data_tensor = torch.tensor(train_df.drop(columns=[target_feature]).values, dtype=torch.float32)
    val_data_tensor = torch.tensor(val_df.drop(columns=[target_feature]).values, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_df.drop(columns=[target_feature]).values, dtype=torch.float32)

    train_targets_tensor = torch.tensor(train_df[target_feature].values, dtype=torch.float32).view(-1, 1)
    val_targets_tensor = torch.tensor(val_df[target_feature].values, dtype=torch.float32).view(-1, 1)
    test_targets_tensor = torch.tensor(test_df[target_feature].values, dtype=torch.float32).view(-1, 1)

    return train_data_tensor, train_targets_tensor, val_data_tensor, val_targets_tensor, test_data_tensor, test_targets_tensor


def main():
    dataset_path = 'datasets/heart_normalized_converted.csv'
    df = pd.read_csv(dataset_path)
    target_feature = 'target'

    while True:
        print("\nSeleziona un'opzione:")
        print("1. Addestrare la rete neurale")
        print("2. Visualizzare le curve di addestramento")
        print("3. Eseguire Grid Search per trovare i migliori iperparametri")
        print("4. Generare il report per validation e test set")
        print("5. Esci")

        scelta = input("Inserisci il numero della tua scelta (1, 2, 3, 4, 5): ")

        if scelta == "1":
            model, history = train_neural_network(df, target_feature, 16, 32, 16, 'adam', 0.001, 32, 100)
            print("\nAddestramento completato!")
            global best_model, best_history
            best_model, best_history = model, history

        elif scelta == "2":
            if 'best_history' not in globals():
                print("\n⚠️ Nessuna rete neurale addestrata! Esegui prima l'opzione 1.")
            else:
                plot_curves_nn(best_history, {'epochs': 100}, metodo="NN")

        elif scelta == "3":
            param_grid = {
                'hidden_units_1': [4, 16, 32],
                'hidden_units_2': [4, 16, 32],
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                'learning_rate': [0.01, 0.001]
            }
            best_model, best_params = grid_search(df, target_feature, param_grid)
            print("\n🏆 Migliori iperparametri trovati:")
            print(best_params)



        elif scelta == "4":

            if 'best_model' not in globals():

                print("\n⚠️ Nessuna rete neurale addestrata! Esegui prima l'opzione 1.")

            else:

                train_data_tensor, train_targets_tensor, val_data_tensor, val_targets_tensor, test_data_tensor, test_targets_tensor = prepare_tensors(
                    df, target_feature)

                generate_classification_report(best_model, val_data_tensor, val_targets_tensor, test_data_tensor,
                                               test_targets_tensor)

        elif scelta == "5":
            print("Uscita dal programma.")
            break

        else:
            print("⚠️ Scelta non valida. Riprova.")

if __name__ == "__main__":
    main()
