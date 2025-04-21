import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils.AbstractRetriever import AbstractRetriever
from utils.AbstractEmbeddingRetriever import AbstractEmbeddingRetriever
from utils.TermEmbedder import TermEmbedder
import torch.nn as nn

class RelationClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.fc(X)
    

    
