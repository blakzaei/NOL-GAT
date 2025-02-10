#-- IMPORT -------------------------------------------------------------------
import sys
from script import load_ds
from script import feature_extraction
from script import split_train_test
from script import graph_construction
from script import nolgat
#------------------------------------------------------------------------------

#-- Initiliaze ----------------------------------------------------------------
K_VALUES = [5,6,7]
DECISION_KEY = 1
EDGE_LIST = 'all'

#------------------------------------------------------------------------------


def main(ds_name, ds_language, amount_labeled, n_iterations, n_epochs):

    load_ds.load_and_prepare_data(ds_name=ds_name)

    feature_extraction.generate_embeddings_using_doc2vec(ds_name=ds_name,
                                                         ds_lang=ds_language)

    split_train_test.split(ds_name=ds_name,
                           train_percent=amount_labeled,
                           number_of_iterations=n_iterations)

    graph_construction.creat_knn_graphs(ds_name=ds_name,
                                        k_values=K_VALUES)

    graph_construction.create_all_hop_edges(ds_name=ds_name,
                                            k_values=K_VALUES)

    nolgat.NOLGAT(ds_name=ds_name,
                  k_values=K_VALUES,
                  edg_list=EDGE_LIST,
                  num_iterations=n_iterations,
                  num_epochs=n_epochs,
                  train_percent=amount_labeled)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
if __name__ == "__main__":
    
    if len(sys.argv) != 6:
        print(sys.argv)
        print('Usage: python main.py <ds_name>, <ds_language>, <amount_labeled> , <n_iterations>, <n_epochs>')
        sys.exit(1)
    
    ds_name = sys.argv[1]
    ds_language = sys.argv[2]
    amount_labeled = float(sys.argv[3])
    n_iterations = int(sys.argv[4])
    n_epochs = int(sys.argv[5])

    
    msg = f'''Start NOL-GAT on Dataset {ds_name}\n
            A KNN graph is created for k = {str(K_VALUES)}\n
            {int(amount_labeled*100)}% of samples are labeled\n
            number of epochs for tarining is {n_epochs}\n
            after {n_iterations} iteration, results will be evaluated.
        '''
    print(msg)
    
    main(ds_name, ds_language, amount_labeled, n_iterations, n_epochs)
###############################################################################



    



