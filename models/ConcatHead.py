import torch
from torch import nn
from torch.nn.init import normal_, constant_


class ConcatHead(nn.Module):

    def __init__(self, feature_dim, modality, num_class, dropout):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.dropout = dropout


        self._add_audiovisual_fc_layer(len(self.modality) * feature_dim, 512)
        self._add_classification_layer(512)


    def _add_classification_layer(self, input_dim):

        std = 0.001
        self.fc_action = nn.Linear(input_dim, self.num_class)
        normal_(self.fc_action.weight, 0, std)
        constant_(self.fc_action.bias, 0)

    def _add_audiovisual_fc_layer(self, input_dim, output_dim):

        self.fc1 = nn.Linear(input_dim, output_dim)
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        std = 0.001
        normal_(self.fc1.weight, 0, std)
        constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        base_out = torch.cat(inputs, dim=1)
        base_out = self.fc1(base_out)
        base_out = self.relu(base_out)
        if self.dropout > 0:
            base_out = self.dropout_layer(base_out)
        base_out = self.fc_action(base_out)
        return base_out
