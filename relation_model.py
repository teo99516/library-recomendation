import networkx as nx
import random
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# Create a training set for the similarity prediction model
# Training features: products of each pair of feature's value
# Training values:  1 when edge exists and 0 when not
# Almost equal number of positive and negative samples
def create_training_set(training_graph, embeddings, libraries, keywords):
    training_features = []
    training_values = []
    for node1, node2, weight in training_graph.edges(data=True):
        training_features.append(np.multiply(embeddings[node1], embeddings[node2]))
        training_values.append(1)

    nodes_names = list(training_graph.nodes())
    negative_values_per_node = len(training_values)/len(libraries)-40
    random.seed(20)

    for node in nodes_names:
        count = 0
        if node[4:] in libraries:
            while count < negative_values_per_node:
                random_node = random.choice(nodes_names)
                if random_node in keywords and not (training_graph.has_edge(node, random_node)):
                    training_features.append(np.multiply(embeddings[node], embeddings[random_node]))
                    training_values.append(0)
                    count = count + 1
    print(Counter(training_values))

    return training_features, training_values


# Train a ML model that predicts if there is a an edge between two nodes
# Return the model and the scaler that was used
def train_relation_model(training_features, training_values):
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)

    # model= LinearSVC(random_state=0, tol=1e-5)
    model = LogisticRegression(random_state=0, penalty="l2")
    train_f = np.matrix(training_features)
    model.fit(train_f, training_values)
    print("Accuracy of Relation model: ", model.score(train_f, training_values))

    return scaler, model
