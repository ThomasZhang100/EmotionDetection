import pandas as pd
import numpy as np
import torch 
from model import SimpleCNN
import matplotlib.pyplot as plt

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
imgs = pd.read_csv('test.csv')
img_str = imgs['pixels'][0]
pixelArray = np.fromstring(img_str, sep=' ', dtype=int)

plt.imshow(np.reshape(pixelArray, (48,48)), interpolation='nearest')
plt.show()

pixels = torch.from_numpy(np.reshape(pixelArray, (1, 1, 48, 48))).float()


torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

model = SimpleCNN() 
model.load_state_dict(torch.load("model.pt"))
model.eval()
print(next(model.children()))


with torch.no_grad():
	output = model(pixels)
	print("Output probabilities:", torch.softmax(output, dim=1))
	print(output)
	_, prediction = torch.max(output, dim=1)


emotion = emotion_dict[prediction.item()]
print("predicted emotion:", emotion)
	
