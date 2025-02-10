#-- IMPORT --------------------------------------------------------------------
import os

import pandas as pd
import numpy as np

import torch
###############################################################################

#-- Function to Create Folders -------------------------------------------------
def create_folder(name=""):
    parent_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_directory_path = os.path.join(parent_folder_path, name)    
    
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)

    return new_directory_path
#------------------------------------------------------------------------------

#-- Function to Create an Empty Dataframe for saving results -------------------
def create_empty_df_for_results(results_file):
    cols_names = ['k',
                  'decision_key',
                  'edge_list',
                  'itr',
                  'percents',
                  'acc',
                  'macro_f1',
                  'micro-f1',
                  'binary-f1',
                  'macro-precision',
                  'micro-precision',
                  'binary-precision',
                  'macro-recall',
                  'micro-recall',
                  'binary-recall']
    
    df_results = pd.DataFrame(columns=cols_names)
    df_results.to_csv(results_file, index=False)
#--------------------------------------------------------------------------





