#-- IMPORT --------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import os
import torch
from utils.util import create_folder
###############################################################################

#-- Initiliaze ----------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#-------------------------------------------------------------------------------

#-- function to create graph ----------------------------------------------------
def creat_knn_graphs(ds_name, k_values):

    #-- create graphs folder --
    data_dir = os.path.join(base_dir, 'data/' + ds_name)
    result_dir = create_folder(os.path.join(data_dir, 'graphs'))

    #-- load data --
    df_file = os.path.join(data_dir, ds_name + '_embeddings.csv')
    df = pd.read_csv(df_file)

    #-- create graphs --
    embedding_columns = [col for col in df.columns if col.startswith('txt_embd')]
    features = df[embedding_columns].values

    for k in k_values:
        print(f'Creating KNN Graph for K = {k} -------------------------')

        knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        knn.fit(features)
        distances, indices = knn.kneighbors(features)

        G = nx.Graph()
        for idx in range(df.shape[0]):
            G.add_node(idx)

        for i in range(df.shape[0]):
            for j in indices[i]:
                if i != j:
                    G.add_edge(i, j)

        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")

        nx.write_graphml(G, os.path.join(result_dir, f'knn_graph_{k}.graphml'))

    print('Creating KNN Graphs: Done :)\n')
#------------------------------------------------------------------------------

#-- Function to Create simulated edges of neighbors in all distances ------------------------
def create_all_hop_edges(ds_name, k_values):
    # -- create results folder --
    data_dir = os.path.join(base_dir, 'data/' + ds_name)
    result_dir = create_folder(os.path.join(data_dir, 'all_hops'))

    # -- set graphs dir --
    graphs_dir = os.path.join(data_dir, 'graphs')

    #-- creating all hops edges --
    for k in k_values:
        print(f'Creating all hops edges for K = {k} -------------------------')

        g_file = os.path.join(graphs_dir, f'knn_graph_{k}.graphml')

        G = nx.read_graphml(g_file)
        G = nx.convert_node_labels_to_integers(G)

        #-- Check if the graph is connected --
        if nx.is_connected(G):
            max_distance = nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            max_distance = nx.diameter(subgraph)

        print(f"Maximum shortest path length: {max_distance}")

        nodes_path_lengths = {}
        for u in G.nodes():
            nodes_path_lengths[u] = nx.single_source_shortest_path_length(G, u)

        for dist in range(2, max_distance + 1):
            # print(f'-- {dist} --------------')

            edges = set()

            for u in G.nodes():
                node_length = nodes_path_lengths[u]
                for v, length in node_length.items():
                    if length == dist:
                        edges.add(tuple(sorted((u, v))))

            edges = list(edges)
            # print(f'edges {dist}: {len(edges)}')

            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            # print('edge_index:', edge_index.shape)
            torch.save(edge_index, os.path.join(result_dir, f'knn_graph_{k}_edge_index_{dist}.pt'))
#------------------------------------------------------------------------------



