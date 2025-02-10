#-- Import -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

import networkx as nx

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import copy
import os

from models import nolgat_net
from utils.util import create_empty_df_for_results
from utils.util import create_folder
#-----------------------------------------------------------------------------------------------


class NOLGAT():
    def __init__(self, ds_name, k_values=[4,5,6], edg_list='all', num_iterations=1,
                 num_epochs=100, train_percent=0.3):

        #-- log --
        print('Running NOL-GAT on dataset ...')

        self.ds_name = ds_name
        self.K_values = k_values
        self.edge_list = edg_list
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.train_percent = train_percent


        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_dir = os.path.join(self.base_dir, 'data/' + self.ds_name)
        self.df_file = os.path.join(self.data_dir, ds_name + '_embeddings.csv')
        self.graphs_dir = os.path.join(self.data_dir, 'graphs')
        self.all_hops_dir = os.path.join(self.data_dir, 'all_hops')
        self.train_indices_dir = os.path.join(self.data_dir, ds_name + '_split')
        self.result_dir = create_folder(os.path.join(self.base_dir, 'results'))
        self.results_file = os.path.join(self.result_dir, 'results.csv')
        create_empty_df_for_results(self.results_file)
        self.df_results = pd.read_csv(self.results_file)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device = {self.device}')

        self.df = None
        self.features = None
        self.edge_index_dict = {}
        self.decision_key = 1
        self.model = None
        self.optimizer = None



        self.load_data()

        self.x = torch.tensor(self.df[self.features].values, dtype=torch.float32)
        self.y = torch.tensor(self.df['label'].values, dtype=torch.long)
        print(f'X:{self.x.shape}\nY:{self.y.shape}')

        self.run()


    #-- function to load data --
    def load_data(self):

        #-- log--
        print(f'loading dataset {self.ds_name} ...')

        self.df = pd.read_csv(self.df_file)

        embedding_columns = [col for col in self.df.columns if col.startswith('txt_embd')]
        self.features = embedding_columns

        #-- log --
        print(f'data loaded: {self.df.shape}')

    #-- function to create edge_index dict from graphs --
    def create_edge_index_dict(self, k, dist_list='all'):
        #-- log --
        print('Creating Edge Indexes for all graphs ...')

        g_file = os.path.join(self.graphs_dir, f'knn_graph_{k}.graphml')

        G = nx.read_graphml(g_file)
        G = nx.convert_node_labels_to_integers(G)

        if dist_list == 'all':
            max_distance = -1
            # -- set max distance --
            if nx.is_connected(G):
                max_distance = nx.diameter(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                max_distance = nx.diameter(subgraph)

            self.edge_index_dict = {}
            dist_list = list(range(2, max_distance + 1))

        # -- 0
        edges = []
        for u in G.nodes():
            edges.append((u, u))

        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        self.edge_index_dict[0] = edge_index

        # -- 1
        edges = list(G.edges)
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        self.edge_index_dict[1] = edge_index

        # -- others
        for i in range(2, len(dist_list) + 2):
            edges_file = os.path.join(self.all_hops_dir, f'knn_graph_{k}_edge_index_{dist_list[i - 2]}.pt' )
            edge_index = torch.load(edges_file)

            self.edge_index_dict[i] = edge_index


    # -- create and initialize model --
    def create_model(self):

        self.model = nolgat_net.NOLGAT_NET(input_size=self.x.size(-1),
                                           hidden_size=128,
                                           output_size=1,
                                           decision_size=self.decision_size,
                                           decision_key=self.decision_key).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=9e-9)



    # -- function to load train indices and create train and test masks --
    def create_masks(self, iteration):
        num_of_nodes = self.x.shape[0]
        train_mask = torch.full((num_of_nodes,), False, dtype=torch.bool)
        train_indices_file = os.path.join(self.train_indices_dir, f'train_indices_{self.train_percent}_{iteration}.pth')
        train_indices = torch.load(train_indices_file)
        train_mask[train_indices] = True
        test_mask = ~train_mask

        return train_mask, test_mask



    #-- function to train model --
    def train(self, train_mask):
        self.model.train()
        self.optimizer.zero_grad()

        logits, decisions_1, decisions_2 = self.model(self.x.to(self.device),
                                                      self.edge_index_dict)
        logits = logits.flatten()

        loss = F.binary_cross_entropy_with_logits(logits[train_mask], (self.y[train_mask] > 0).float())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float().squeeze()

        accuracy = accuracy_score(self.y[train_mask].cpu(), preds[train_mask].int().cpu())

        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy, decisions_1, decisions_2


    #-- function to evaluate model --
    def evaluate(self, test_mask, model):
        model.eval()
        with torch.no_grad():
            logits, decisions_1, decisions_2 = model(self.x.to(self.device),
                                                     self.edge_index_dict)
            logits = logits.flatten()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().squeeze()

            loss = F.binary_cross_entropy_with_logits(logits, (self.y > 0).float())
            accuracy = accuracy_score(self.y[test_mask].cpu(), preds[test_mask].int().cpu())

        return loss, accuracy, decisions_1, decisions_2, preds


    #-- function to calculate all metrics --
    def fina_evaluate_and_save(self, y_preds, y_true, iter, k, train_percent, test_mask):

        real_labels = y_true[test_mask].cpu()
        pred_results = y_preds[test_mask].int().cpu()

        acc_val = accuracy_score(real_labels, pred_results)

        macro_f1_val = f1_score(real_labels, pred_results, average='macro')
        micro_f1_val = f1_score(real_labels, pred_results, average='micro')
        binary_f1_val = f1_score(real_labels, pred_results)

        macro_pr_val = precision_score(real_labels, pred_results, average='macro')
        micro_pr_val = precision_score(real_labels, pred_results, average='micro')
        binary_pr_val = precision_score(real_labels, pred_results)

        macro_re_val = recall_score(real_labels, pred_results, average='macro')
        micro_re_val = recall_score(real_labels, pred_results, average='micro')
        binary_re_val = recall_score(real_labels, pred_results)

        # -- Save reults to cvs full results ------------------------------------
        df_results = pd.read_csv(self.results_file)
        results = {'k': k,
                   'decision_key': self.decision_key,
                   'edge_list': str(self.edge_list),
                   'itr': iter,
                   'percents': train_percent,
                   'acc': acc_val,
                   'macro_f1': macro_f1_val,
                   'micro-f1': micro_f1_val,
                   'binary-f1': binary_f1_val,
                   'macro-precision': macro_pr_val,
                   'micro-precision': micro_pr_val,
                   'binary-precision': binary_pr_val,
                   'macro-recall': macro_re_val,
                   'micro-recall': micro_re_val,
                   'binary-recall': binary_re_val}

        new_df = pd.DataFrame(results, index=[0])
        df_results = pd.concat([df_results, new_df], ignore_index=True)
        df_results.to_csv(self.results_file, index=False)
        return


    def run(self):
        for k in self.K_values:
            print(f'\n\n\n K = {k} =======================================')

            #-- log --
            print('loading graph ...')

            g_file = os.path.join(self.graphs_dir, f'knn_graph_{k}.graphml')
            G = nx.read_graphml(g_file)
            G = nx.convert_node_labels_to_integers(G)

            print("\tNumber of nodes:", nx.number_of_nodes(G))
            print("\tNumber of edges:", nx.number_of_edges(G))


            self.create_edge_index_dict(k, dist_list=self.edge_list)
            self.decision_size = len(self.edge_index_dict)

            test_accs = []
            for itr in range(1, self.num_iterations + 1):
                print(f'Iteration {itr} ....................................')

                train_mask, test_mask = self.create_masks(itr)
                self.create_model()

                best_acc = 0
                best_epoch = 0
                best_model = None

                for epoch in range(1, self.num_epochs + 1):
                    train_loss, train_acc, decisions_1_tr, decisions_2_tr = self.train(train_mask)
                    val_loss, val_acc, decisions_1_tst, decisions_2_tst, preds = self.evaluate(test_mask, self.model)

                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_epoch = epoch
                        best_model = copy.deepcopy(self.model)

                    if epoch == 1 or epoch % 10 == 0 or epoch == self.num_epochs:
                        log = f'Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}'
                        print(log)


                test_loss, test_acc, decisions_1, decisions_2, preds = self.evaluate(test_mask , best_model)

                test_accs.append(best_acc)

                self.fina_evaluate_and_save(y_preds=preds,
                                            y_true=self.y,
                                            iter=itr,
                                            k=k,
                                            train_percent=self.train_percent,
                                            test_mask=test_mask)


            print('AVG TEST ACC = ', np.round(sum(test_accs) / len(test_accs), 4))


        print('\nRunning DHGAT on LIAR: DONE :)\n')









