import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base import BaseModel
# from bio_embeddings.embed import SeqVecEmbedder, UniRepEmbedder

class SeqVecEmbedding(nn.Module):
    """ SeqVec embedding layer module """

    def __init__(self):
        super(SeqVecEmbedding, self).__init__()
        # self.embedder = SeqVecEmbedder(weights_file=weights_file, options_file=options_file, max_amino_acids = max_amino_acids)

    def forward(self, embedding: torch.tensor) -> torch.tensor:
        # generator = self.embedder.embed_batch(batch_data)
        # # shape=batch_size, 1024
        # embedding = torch.stack([torch.tensor(seq.sum(axis=0).mean(axis=0)) for seq in generator], dim=0)
        
        # convert nan to num 0.0
        embedding[embedding != embedding] = 0.0

        # # cleanup
        # del batch_tokens
        # torch.cuda.empty_cache()

        # return embedding # batch_size, 1024
        return embedding # batch_size, 1024

class Baseline(BaseModel):
    def __init__(self, n_hidden, **kwargs):
        """ Constructor
        Args:
            init_n_channels: size of the incoming feature vector
            n_hidden: amount of hidden neurons
            n_hidden1: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        super().__init__()

        # SeqVec block
        self.embedding = SeqVecEmbedding()

        # loss function will do log_softmax inside, no need to add a softmax layer here.
        # but remember to convert the output to probability by using: torch.exp(F.log_softmax(input, dim=1))
        self.net = nn.Sequential(*[
        nn.Linear(1024, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 128),
        nn.ReLU(),
        nn.Linear(128,2)])
        
    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """
        """
        x: batch_tokens (tensor)
        mask: valid_lengths of batch_tokens (tensor)
        """
        x = self.embedding(x)
        # # average residual embeddings to seq embeddings
        # x = x.mean(axis=1)
        # output shape=(batch_size, seq_len, 2)
        x = self.net(x)
        return x
