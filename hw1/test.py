import dataset as cds
import network as ntw
import glob
import numpy as np

controls = np.load('output/observations/controls.npy')
controls = controls[controls[:,0] > 29722, :]
controls = controls[controls[:,0] < 29947, :]
print(controls[:, 1] > 0.01)
