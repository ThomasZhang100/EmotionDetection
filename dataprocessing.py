
import os
print(os.getcwd())

import numpy as np
from torch.utils import data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class set_up_data(data.Dataset):
    def __init__(self, df):
        self.data = df.values
        self.n_samples = len(self.data)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        target = self.data[index,0]
        pixelString = self.data[index,1]
        pixelArray = np.fromstring(pixelString, sep=' ', dtype=int)
        pixels = torch.from_numpy(np.reshape(pixelArray, (1, 48, 48))).float()
        return pixels, target




    
df = pd.read_csv("train.csv")
train, test = train_test_split(df, test_size=0.1)
#print(df.values[0,0])

train_data = set_up_data(train)
test_data = set_up_data(test)

#print(len(train_data))

train_loader = data.DataLoader(train_data, batch_size = 200, num_workers=0, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size = 200, num_workers=0, shuffle=True)
#%%
if __name__ == "__main__":
    data = set_up_data(df)
    target, pixels = data.__getitem__(0)
    print(pixels)
    plt.imshow(np.reshape(pixels.numpy(), (48,48)), interpolation='nearest')
    plt.show()


# %%
