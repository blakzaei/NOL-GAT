#-- IMPORT --------------------------------------------------------------------
import torch
import pandas as pd
import random
import os

from utils.util import create_folder
###############################################################################

#-- Initiliaze ----------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#------------------------------------------------------------------------------

#-- Function to split data to train and test -----------------------------------
def split(ds_name, train_percent, number_of_iterations):
    # -- log --
    print(f'Spliting Data to {int(train_percent*100)}% as labeled and {int((1-train_percent)*100)}% as unlabeled ...')

    #-- load data --
    data_dir = os.path.join(base_dir, 'data/' + ds_name)
    df_file = os.path.join(data_dir, ds_name + '.csv')
    df = pd.read_csv(df_file)

    # -- create split folder --
    result_dir = create_folder(os.path.join(data_dir, ds_name + '_split'))

    #-- split ds into train and test --
    all_indices = df.index.tolist()
    ds_size = len(df)
    num_of_train_samples = int(train_percent * ds_size)
    print(f'number of all samples: {ds_size}\nnumber of train samples: {num_of_train_samples}')

    # -- select train_percent% of nodes randomly --
    for itr in range(1, number_of_iterations + 1):
        print(f'-- iteration {itr} -----------------------------------')

        train_indices = torch.tensor(random.sample(all_indices, num_of_train_samples))
        torch.save(train_indices, os.path.join(result_dir, f'train_indices_{train_percent}_{itr}.pth'))

        labels_in_train = df.loc[train_indices.numpy(), 'label']
        count_label_0 = (labels_in_train == 0).sum()
        count_label_1 = (labels_in_train == 1).sum()

        print(f'\tNumber of label 0 = real in train indices: {count_label_0}')
        print(f'\tNumber of label 1 = fake in train indices: {count_label_1}')

    print('Spliting Data: DONE :)\n')
#------------------------------------------------------------------------------


