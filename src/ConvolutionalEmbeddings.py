import torch
from torch import nn
import torch.nn.functional as F

#TODO Create embedding model for downsampling and embedding .wav files (Ryan)


class ResidualDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()
        #initial embed
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        #reduce internal covariate shift
        self.bn1 = nn.BatchNorm1d(out_channels)
        #downsample embed using stride
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        '''
            This block takes waveform input, embeds, normalizes, downsamples, and reembeds
            returns embeded waveform as a tensor with a size reduced by a factor of stride
        '''
        output = self.conv1(x)
        output = self.bn1(output)
        #Relu + x for a residual connection to help gradient flow.
        # AKA network only has to produce the diff from input to output 
        output = self.relu(output) + x
        output = self.conv2(output)
        return output


class DownsamplingNetwork(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, in_channels=1,
                 initial_mean_pooling_kernel_size = 2, strides=[6,6,8,4,2]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mean_pooling = nn.MaxPool1d(kernel_size=initial_mean_pooling_kernel_size)

        for i in range(len(strides)):
            self.layers.append(
                ResidualDownSampleBlock(
                    hidden_dim if i>0 else in_channels,
                    hidden_dim,
                    strides[i],
                    kernel_size=8
                )
            )
        self.final_conv = nn.Conv1d(hidden_dim, embedding_dim, kernel_size=4, padding="same")

    def forward(self, x):
        x=self.mean_pooling(x)
        i=1
        for layer in self.layers:
            x = layer(x)
            i+=1
        x = self.final_conv(x)
        x =  x.transpose(1,2)
        return x
        


if __name__ == "__main__":
    batch_size = 2
    input_embed_dim = 1
    seq_len = 10000
    output_embed_dim = 32
    hidden_dim = 16
    #Each stride creates a downsample layer reducing size by a factor of the stride value
    strides = [2, 4, 6]
    #This divides the size for some reason so watch out when testing
    initial_mean_pooling_kernel_size = 2
    DSN = DownsamplingNetwork(
        embedding_dim=output_embed_dim,
        hidden_dim=hidden_dim,
        in_channels=input_embed_dim,
        initial_mean_pooling_kernel_size=initial_mean_pooling_kernel_size,
        strides=strides

    )
    x = torch.randn(batch_size, 1, seq_len)
    print(DSN(x).shape)












