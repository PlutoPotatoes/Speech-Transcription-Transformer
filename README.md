# Speech-Transcription-Transformer

This model utilizes convolutional downsampling, vector quantization, and transformer attention to convert waveform audio files to audio transcriptions.

#Dependencies

The training process for this model relies on the following packages:
 - torch
 - scipy
 - tokenizer
 - datasets

All necessary packages are available on pip. Datasets versioning may vary depending on your torch version.

# Functionality

The following layers are explained in the order the model uses them. 

1. ConvolutionalEmbeddings.py
    - Downsamples waveform files to build initial embeddings
2. ResidualQuantizer.py 
    - Builds codebooks of vectors based on previously seen samples to adjust new sample inputs
3. AttentionEmbeddings.py
    - uses transformer blocks to add attention to the embeddings

The model then uses a single linear layer, softmaxes over 29 character class probabilities, and uses CTC loss to adjust the gradient.

# Usage

Use main() in train.py to train a model and load it with TranscriberModel.load('Model name'). The current code is set up to run on the devices CPU but with minor adjustments it could be altered to use a CUDA enabled GPU device for both training and testing