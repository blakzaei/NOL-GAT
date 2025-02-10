#-- Import -------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F

from models.nolgat_layer import NOLGAT_Layer
#-----------------------------------------------------------------------------------------------

class NOLGAT_NET(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, decision_size, decision_key):
        super().__init__()

        self.nol_layer_1 = NOLGAT_Layer(input_size, hidden_size, decision_size, decision_key)
        self.nol_layer_2 = NOLGAT_Layer(hidden_size, hidden_size // 2, decision_size, decision_key)

        self.fc1 = torch.nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc2 = torch.nn.Linear(hidden_size // 4, output_size)

    def forward(self, x, edge_index_dict):
        x, decisions_1 = self.nol_layer_1(x, edge_index_dict)
        x = x.relu()
        x, decisions_2 = self.nol_layer_2(x, edge_index_dict)
        x = x.relu()

        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x, decisions_1, decisions_2
# -----------------------------------------------------------------------------------------------