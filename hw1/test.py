import dataset as cds
import network as ntw
import glob
import numpy as np

data = cds.CarlaDataset('output/train/')
start = 352492

network = ntw.ClassificationNetwork()

actions = np.empty(3)
for i in range(data.__len__()):
    pic, action = data.__getitem__(i)
    print(network.actions_to_classes(action[[1,2,3]]))
