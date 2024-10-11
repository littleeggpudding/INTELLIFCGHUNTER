import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier  # 或 RandomForestRegressor
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from sklearn.linear_model import PassiveAggressiveClassifier
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from type.FCG_malscan import FCG_malscan as FCG


def train_model(vector, label, model_name, model_output_path, type=''):
    """
    This function trains various machine learning models based on the provided model name.
    It supports K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), Support Vector Machine (SVM),
    RandomForest, PassiveAggressiveClassifier (PA1), and AdaBoost. The trained model is saved to the specified path.

    Args:
        vector (numpy array): Feature vectors for training the model.
        label (numpy array): Labels corresponding to the training data.
        model_name (str): The name of the model to be trained. Options include 'KNN_1', 'KNN_3', 'KNN_5', 'KNN_10',
                          'MLP', 'SVM', 'RandomForest', 'PA1', 'AdaBoost'.
        model_output_path (str): Path where the trained model will be saved.
        type (str): Additional type to distinguish model files (optional).

    Returns:
        str: Path to the saved model file.
    """

    try:
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        print(model_output_path)
        if model_name == 'KNN_1':
            knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=2)
            knn1.fit(vector, label)
            joblib.dump(knn1, model_output_path + type + 'knn_1.pkl')
            print('knn_1.pkl saved')
            return model_output_path + 'knn_1.pkl'

        elif model_name == 'KNN_3':
            knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs=2)
            knn3.fit(vector, label)
            joblib.dump(knn3, model_output_path + type + 'knn_3.pkl')
            return model_output_path + 'knn_3.pkl'

        elif model_name == 'KNN_5':
            knn5 = KNeighborsClassifier(n_neighbors=5, n_jobs=2)
            knn5.fit(vector, label)
            joblib.dump(knn5, model_output_path + type + 'knn_5.pkl')
            return model_output_path + 'knn_5.pkl'

        elif model_name == 'KNN_10':
            knn10 = KNeighborsClassifier(n_neighbors=10, n_jobs=2)
            knn10.fit(vector, label)
            joblib.dump(knn10, model_output_path + type + 'knn_10.pkl')
            return model_output_path + 'knn_10.pkl'

        elif model_name == 'MLP':
            MLP = Sequential([
                Dense(128, activation='relu', input_shape=(vector.shape[1],)),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            MLP.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            checkpoint = ModelCheckpoint(model_output_path + type + 'MLP.h5',
                                         monitor='loss',
                                         save_best_only=True,
                                         verbose=1)
            MLP.fit(vector, label,
                    batch_size=16,
                    epochs=100,
                    callbacks=[checkpoint])
            return model_output_path + type + 'MLP.h5'

        elif model_name == 'SVM':
            svmModel = svm.SVC(kernel='rbf', C=1, gamma='scale', probability=True)
            svmModel.fit(vector, label)
            joblib.dump(svmModel, model_output_path + type + 'SVM.pkl')
            return model_output_path + 'SVM.pkl'

        elif model_name == 'RandomForest':
            rf = RandomForestClassifier(max_depth=3, random_state=0)
            rf.fit(vector, label)
            joblib.dump(rf, model_output_path + type + 'RandomForest.pkl')
            return model_output_path + type + 'RandomForest.pkl'

        elif model_name == 'PA1':
            pa1 = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
            pa1.fit(vector, label)
            joblib.dump(pa1, model_output_path + type + 'PA1.pkl')
            return model_output_path + type + 'PA1.pkl'

        elif model_name == 'AdaBoost':
            ada = AdaBoostClassifier(n_estimators=100, random_state=0)
            ada.fit(vector, label)
            joblib.dump(ada, model_output_path + type + 'AdaBoost.pkl')
            return model_output_path + type + 'AdaBoost.pkl'

        else:
            print('Wrong model name!')
            return None

    except Exception as e:
        print(f"An error occurred: {e}")



def train_substitute_Model(vector, original_model, model_name, ori_model_name, model_output_path, type=''):
    try:
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        if model_name == 'MLP':
            MLP = Sequential([
                Dense(128, activation='relu', input_shape=(vector.shape[1],)),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            MLP.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            checkpoint = ModelCheckpoint(model_output_path + type + 'MLP_' + ori_model_name + '.h5',
                                         monitor='loss',
                                         save_best_only=True,
                                         verbose=1)
            label = original_model.predict(vector)
            MLP.fit(vector, label,
                    batch_size=16,
                    epochs=100,
                    callbacks=[checkpoint])
            return model_output_path + type + 'MLP_' + ori_model_name + '.h5'
        else:
            print('Wrong model name!')
            return None

    except Exception as e:
        print(f"An error occurred: {e}")





class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, 128)
        self.conv5 = GCNConv(128, 128)
        self.fc = nn.Linear(128, num_classes)  # Adaptive binary-classification output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_index = edge_index.t()
        # 5 layers GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Global average pooling to aggregate graph features
        x = global_mean_pool(x, batch)

        # Classification is done through a fully connected layer
        x = self.fc(x)

        return x

def obtain_gcn_feature(file, label):
    fcg = FCG(file, label, 0)
    edge_index = []
    degree_features = []
    for edge in fcg.edges:
        edge_index.append([edge[0], edge[1]])
    for node in fcg.nodes:
        in_degree = fcg.current_call_graph.in_degree(node)
        out_degree = fcg.current_call_graph.out_degree(node)
        degree_features.append([in_degree, out_degree])
    degree_features = torch.tensor(degree_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=degree_features, edge_index=edge_index, y=y)
    return data

def merge_data(edge_list, node_features_list):
    cumulative_nodes = 0
    edge_index_list = []
    x_list = []
    batch_list = []

    for i, (edges, features) in enumerate(zip(edge_list, node_features_list)):
        num_nodes = features.size(0)
        x_list.append(features)
        batch_list.append(torch.full((num_nodes,), i, dtype=torch.long))

        edge_index_list.append(edges + cumulative_nodes)
        cumulative_nodes += num_nodes

    return torch.cat(edge_index_list, dim=1), torch.cat(x_list, dim=0), torch.cat(batch_list, dim=0)


def train_gcn_model(data_list, save_path, epoch = 200):
    """
    Trains a GCN (Graph Convolutional Network) model on the provided graph data and saves the trained model.

    Args:
    - data_list: A list of PyTorch Geometric `Data` objects representing graph-structured data.
      Each `Data` object typically contains the following:
        - x: Node feature matrix, where each row represents the features of a node.
        - edge_index: Graph connectivity in COO format, i.e., a 2D tensor with shape [2, num_edges].
        - y: Labels for nodes or graphs, depending on the task.
    - save_path: Path to save the trained model after training.

    Training Details:
    - Assumes each node has 2 features (num_node_features=2).
    - Binary classification task (num_classes=2).
    - Uses Adam optimizer and CrossEntropyLoss.
    - Model is trained for 30 epochs, using a single graph per batch (batch_size=1).
    - Trained on GPU if available, otherwise on CPU.
    """

    loader = DataLoader(data_list, batch_size=1, shuffle=True)

    # Initialize the GCN model with 2 node features and 2 output classes (binary classification)
    model = GCN(num_node_features=2, num_classes=2)

    # Define the loss function as cross-entropy for classification tasks
    criterion = torch.nn.CrossEntropyLoss()

    # Use Adam optimizer for weight updates
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set the model to training mode
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(epoch):
        for data in loader:
            data = data.to(device)  # Move graph data to the selected device
            optimizer.zero_grad()  # Clear previous gradients
            out = model(data)  # Forward pass through the GCN
            loss = criterion(out, data.y)  # Compute the loss based on predictions and true labels
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update model weights

        # Print loss for the current epoch
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the trained model's state dictionary to the specified save path
    torch.save(model.state_dict(), f'{save_path}/gcn_model.pth')


def test_gcn_model(model, test_data_list, model_path):
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Set up device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    loader = DataLoader(test_data_list, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during testing
        for data in loader:
            data = data.to(device)  # Move data to the selected device
            out = model(data)  # Forward pass through the GCN model
            pred = out.argmax(dim=1)  # Get the predicted class (argmax over output logits)
            correct += pred.eq(data.y).sum().item()  # Compare prediction with true label
            total += data.y.size(0)  # Total number of nodes/graphs (depending on task)

    # Calculate and print accuracy
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

    
    
