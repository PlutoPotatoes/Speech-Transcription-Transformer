import torch
import torch.nn as nn

#TODO Create Residual Quantizer to map embeddings to tokens



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().init()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape
        flat_x = x.reshape(batch_size * sequence_length, embedding_dim)

        #compute distances 
        distances = torch.cdist(flat_x, self.embedding.weight, p=2)

        #encode the closest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(
            batch_size, sequence_length, embedding_dim
        )

        #update all x gradient throughout the whole network up until now and calculate loss
        e_latent_loss = torch.mean((quantized.detatch()-x)**2)
        #update quantized layers
        q_latent_loss = torch.mean((quantized-x.detach())**2)

        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        #reattach the quantized gradient to x since argmin stops its flow
        quantized = x + (quantized-x).detach()
        return quantized, loss



class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        super().__init__()
        self.codebooks = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim)
            for i in range(num_codebooks)
        ])
    def forward(self, x):
        out = 0
        total_loss = 0
        for codebook in self.codebooks:
            this_output, this_loss = codebook(x)
            x = x-this_output
            out = out+this_output
            total_loss+=this_loss
        return out, total_loss
    

