import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--frame", help='cut off any observations before this frame')
parser.add_argument('-r', "--run", help='file path of the run to trim')
args = parser.parse_args()

controls = np.load(args.run+'/controls.npy')
mask_array = controls[:, 0] < int(args.frame)
np.save(args.run+'/controls_trim.npy', controls[~mask_array,1:4])
print(np.load(args.run+'/controls_trim.npy'))
