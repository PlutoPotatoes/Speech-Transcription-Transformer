import torch
import dataset
import Levenshtein
import numpy as np
from model import TranscribeModel

'''
11.23 = 20.72

12.1 = 36.64
12.2 = 38.11



'''
modelPath = 'models/test12.2_final.pth'

def test():
    #model = torch.load('models/test11.20_final.pth', weights_only=False, map_location=torch.device('cpu'))
    model = TranscribeModel.load(modelPath)
    vocab = ['□', 'A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', ' ', '\'']

    dataloader = dataset.get_dataset(batch_size=1, num_examples=100)
    scores = []
    for i, x in enumerate(dataloader):
        audio = x["audio"]
        text = x["text"]
        with torch.no_grad():
            output, vq_loss = model(audio)
            str = ''
            letters = torch.argmax(output[0], dim=1)
            for letter in letters:
                letter = vocab[letter.item()]
                if(letter != '□'):
                    str = str + letter
            score = Levenshtein.distance(str, text)
            scores.append(score)
            print(str)
            print(text)
    print(scores)
    print(f"Mean Score = {np.mean(scores)}")
                



if __name__ == "__main__":
    test()