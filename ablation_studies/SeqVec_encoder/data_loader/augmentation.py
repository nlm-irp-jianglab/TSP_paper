import torch
from bio_embeddings.embed import SeqVecEmbedder, UniRepEmbedder

class SeqVecTokenize(object):
    """ Tokenizes a sequence for SeqVec model input """
    def __init__(self, weights_file, options_file, max_amino_acids: int = 2000):
        self.embedder = SeqVecEmbedder(weights_file=weights_file, options_file=options_file, max_amino_acids = max_amino_acids)
        
    def __call__(self, x):
        # batch_labels = x[0]
        generator = self.embedder.embed_batch(x[1])
        batch_data = torch.stack([torch.tensor(seq.sum(axis=0).mean(axis=0)) for seq in generator], dim=0)
        return batch_data