#-- Import -------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
#-----------------------------------------------------------------------------------------------

class NOLGAT_Layer(torch.nn.Module):
    def __init__(self, input_size, output_size, decision_size, decision_key):
        super().__init__()

        self.gat = GATv2Conv(input_size,
                             output_size,
                             heads=1,
                             dropout=0.5)

        self.decision = GATv2Conv(input_size,
                                  decision_size,
                                  heads=1,
                                  dropout=0.5)

        self.n_decisions = decision_size
        self.input_size = input_size
        self.output_size = output_size
        self.decision_key = decision_key
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, edge_index_dict):
        decision_logits = self.decision(x, edge_index_dict[self.decision_key].to(self.device))
        decisions = F.gumbel_softmax(decision_logits, tau=0.5, hard=True)

        out = torch.zeros(x.shape[0], self.output_size).to(self.device)

        for i in range(self.n_decisions):
            out += self.gat(x, edge_index_dict[i].to(self.device)) * decisions[:, i:i + 1]
        return out, decisions
# -----------------------------------------------------------------------------------------------