"""funcGNN class and runner."""

import glob
import torch
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss
from sklearn import metrics
import concurrent.futures as cf
import time

class funcGNN(torch.nn.Module):
    """
    funGNN: A Graph Neural Network Approach to Program Similarity 
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        :param embedding_size: word vector embedding size (100/200/300).
        """
        super(funcGNN, self).__init__()
        self.args = args
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = SAGEConv(self.args.embedding_size, self.args.filters_1, normalize = True)
        self.convolution_2 = SAGEConv(self.args.filters_1, self.args.filters_2, normalize = True)
        self.convolution_3 = SAGEConv(self.args.filters_2, self.args.filters_3, normalize = True)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)

        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score

class funcGNNTrainer(object):
    """
    funcGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
        """
        Creating a funcGNN.
        """
        self.model = funcGNN(self.args)

    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph+self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        edges_1 = data["graph_1"] + [[y,x] for x,y in data["graph_1"]]
        edges_2 = data["graph_2"] + [[y,x] for x,y in data["graph_2"]]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1, features_2 = data["labels_1"], data["labels_2"]
        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))
        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2
        new_data["features_1"] = features_1
        new_data["features_2"] = features_2
        norm_ged = data["ged"]
        new_data["target"] = torch.from_numpy(np.array([norm_ged]).reshape(1,1)).float()
        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses =0
        for graph_pair in batch:
            data = process_pair(graph_pair)
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(prediction, target)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def get_train_baseline_error(self):
        """
        Calculates the baseline error of the training data
        """
        self.train_ground_truth = []
        for graph_pair in tqdm(self.training_graphs):
            data = process_pair(graph_pair)
            self.train_ground_truth.append(data["ged"])
        norm_ged_mean = np.mean(self.train_ground_truth)
        # base_train_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.train_ground_truth])
        # print("\nBaseline Training error: " + str(round(base_train_error, 5)))

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        epoch_counter =0
        loss =0
        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            self.loss_sum = 0
            self.epoch_loss =0
            self.node_processed = 0
            batches = self.create_batches() # now batches are just json file names. 
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                self.epoch_loss = self.epoch_loss + self.process_batch(batch)
                self.node_processed = self.node_processed + len(batch)
                loss = self.epoch_loss/self.node_processed
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 6))
            with open("./outputFiles/test/train_error_graph.txt", "a") as train_error_writer:    
                train_error_writer.write(str(epoch_counter) + ',' + str(round(loss, 6)) + '\n')
            
            torch.save(self.model.state_dict(), './outputFiles/test/model_state_{}.pth'.format(epoch))
            epoch_counter += 1
            self.score(epoch_counter)

    def score(self, epoch_counter):
            """
            Scoring on the test set.
            """
            print("\n\nModel evaluation.\n")
            start_time = time.time()
            self.model.eval()
            self.scores = []
            self.ground_truth = []
            self.predictions = []
            for test_graph_pair in tqdm(self.testing_graphs):
                data = process_pair(test_graph_pair)
                self.ground_truth.append(data['ged'])
                data = self.transfer_to_torch(data)
                target = data["target"]
                prediction = self.model(data)
                # try:
                self.scores.append(calculate_loss(prediction, target))
                self.predictions.append(prediction)
                # except:
                #     print(test_graph_pair)
            print("--- %s seconds ---" % (time.time() - start_time))
            model_error = self.print_evaluation()
            with open("./outputFiles/test/test_error_graph.txt", "a") as test_error_writer:
                test_error_writer.write(str(epoch_counter) + ',' + str(model_error)+ '\n')
            test_error_writer.close()
            # self.calculate_f1_score()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        return str(round(model_error, 6))

    def runParallelCode(self, pairList):
        with cf.ProcessPoolExecutor(max_workers =5) as executor:
            results = [executor.submit(self.load_model_parallel, files) for files in pairList]

    def load_model(self):
        print("\nSerial Execution of funcGNN from pretrained model")
        start_time = time.time()
        self.model = funcGNN(self.args)
        self.model.load_state_dict(torch.load('./outputFiles/test/model_state.pth'))
        self.model.eval()
        self.scores = []
        self.ground_truth = [] 
        self.predictions = []

        for test_graph_pair in tqdm(self.testing_graphs):
            data = process_pair(test_graph_pair)
            self.ground_truth.append(data["ged"])
            f_res = open("./outputFiles/test/result_{}.txt".format(self.args.test_arch), "a") 
            f_res.write(str(data["ged"]) +  " " + test_graph_pair + " : ")
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            self.scores.append(calculate_loss(prediction, target))
            self.predictions.append(prediction)
            f_res.write(str(prediction) + '\n')
    
        print("--- %s seconds ---" % (time.time() - start_time))
        self.ROC()

    def start_parallel(self):
        print("Parallel Execution of funcGNN from pretrained model")
        start_time = time.time()
        self.graph_pairList = []
        self.scores = []
        self.ground_truth = []
        self.model = funcGNN(self.args)
        self.model.eval()

        for test_graph_pair in tqdm(self.testing_graphs):
            self.graph_pairList.append(test_graph_pair)
        self.runParallelCode(self.graph_pairList)
        print("--- %s seconds ---" % (time.time() - start_time))

    def calculate_accuracy(predictions, ground_truth, threshold=0.5):
        """
        Calculate the accuracy of predictions compared to the ground truth.
        
        :param predictions: List of predicted values.
        :param ground_truth: List of true values.
        :param threshold: Threshold to classify predictions as positive or negative (default is 0.5).
        :return: Accuracy score.
        """
        predicted_labels = [1 if pred >= threshold else 0 for pred in predictions]
        true_labels = [1 if gt >= threshold else 0 for gt in ground_truth]
        correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))
        
        accuracy = correct_predictions / len(ground_truth)      
        return accuracy

    def ROC(self):        
        print("Calculating similarity scores, ROC...")
        ends = []
        arch = self.args.test_arch
        for each in self.predictions:
            each = each.detach().numpy()
            each = np.ndarray.tolist(each)
            each = each[0]
            each = each[0]
            ends.append(each)
        fpr1, tpr1, thresholds1 = metrics.roc_curve(self.ground_truth, ends, pos_label=1)
        auc1 = metrics.auc(fpr1, tpr1)
        print("AUC avg cos: %f" % auc1)

        
        plt.plot(fpr1, tpr1, label="Testing {}, AUC=%f".format(arch) %auc1)
        plt.legend(loc = 'lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        with open('./outputFiles/test/{}_scores.pickle'.format(arch), 'wb') as f:
            pickle.dump(self.scores, f)
        with open('./outputFiles/test/{}_gt.pickle'.format(arch), 'wb') as f:
            pickle.dump(self.ground_truth, f)

        with open('./outputFiles/test/{}_fpr1.pickle'.format(arch), 'wb') as f:
            pickle.dump(fpr1, f)
        with open('./outputFiles/test/{}_tpr1.pickle'.format(arch), 'wb') as f:
            pickle.dump(tpr1, f)

        plt.savefig('./outputFiles/test/roc_{}.png'.format(arch))        
        optimal_threshold = thresholds1[np.argmax(tpr1 - fpr1)]
        predicted_labels = [1 if pred >= optimal_threshold else 0 for pred in ends]
        true_labels = [1 if gt >= optimal_threshold else 0 for gt in self.ground_truth]
        correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))
        
        accuracy = correct_predictions / len(self.ground_truth)      
        print(f"Accuracy Score: {accuracy:.4f}")
        with open(f"./outputFiles/test/acc_{arch}.txt", "a") as acc_writer:
            acc_writer.write(f"Acc Score = {accuracy:.4f}\n")
