#-- IMPORT --------------------------------------------------------------------
import pandas as pd
import os

from utils.util import create_folder
###############################################################################

#-- Function to load and prepare LIAR ds --------------------------------------
def load_and_prepare_data(ds_name):
    parent_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_dir = os.path.join(parent_folder_path, 'data/'+ds_name)
    input_ds_file = os.path.join(ds_dir, ds_name + '.tsv')

    output_ds_file = os.path.join(ds_dir, ds_name + '.csv')

    if not os.path.isfile(input_ds_file):
        print(f"dataset {ds_name} not found!\n")
        return

    #-- load ds --
    print(f'loading dataset {ds_name} ...')
    df = pd.read_csv(input_ds_file, sep='\t', header=None)
    df.columns = ['id', 'txt', 'label']

    # -- Shuffle df --
    df = df.sample(frac=1).reset_index(drop=True)

    # -- replace label -1(reals) with 0--
    df['label'] = df['label'].replace(-1, 0)

    # -- 1 is fake and 0 is real --
    n_fakes = df['label'].value_counts()[1]
    n_reals = df['label'].value_counts()[0]

    print('number of all news=%d\nnumber of fake news=%d \nnumber of real news=%d' % (df.shape[0],
                                                                                      n_fakes, n_reals))

    #-- save --
    print(f'Saving dataset {ds_name} to CSV file ...')
    df.to_csv(output_ds_file, index=False)

    print('load_and_prepare_data: DONE :)\n')
#------------------------------------------------------------------------------
