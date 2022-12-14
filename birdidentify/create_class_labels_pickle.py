import os
import pickle
import numpy as np
basedir = os.path.abspath(os.path.dirname(__file__))

classes=[]
for i, fname in enumerate(os.listdir(os.path.join(basedir, "static", "train"))):
    classes += [fname]
classes=np.array(classes)
print(classes)

with open('classlabels.pkl', 'wb') as fh:
   pickle.dump(classes, fh)

