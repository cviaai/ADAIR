import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time
import os, sys, random
from pathlib import Path
import pandas as pd

df = pd.read_csv("./file.csv")

asd = np.array(df.to_numpy())

single_run = []
for i in range(len(asd)):
    single_run.append(asd[i, 1])

print(len(single_run))
print(single_run)
plt.plot(single_run, label="Greedy Softmax Q-window")
plt.xlabel("Episodes")
plt.ylabel("Normalized\nsteps\nduring\nepisode", labelpad=40)
plt.xlim(0,len(single_run))
plt.legend()
plt.tight_layout()
#plt.savefig('plots/plots_'+str(episode)+'.png')
plt.gcf().clear()
plt.show()