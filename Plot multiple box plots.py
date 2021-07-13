
import numpy as np
import matplotlib.pyplot as plt
boxes = []   # empty list
set1 = np.random.randn(100)
boxes.append(set1)
set2 = np.random.randn(100)
boxes.append(set2)
set3 = np.random.randn(100)
boxes.append(set3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(boxes)
ax.set_title('PlotMultipleBoxes')
ax.set_xlabel('Gaussian Set1 (1), Set2 (2), set3 (3)')
ax.set_ylabel('vals')
plt.show() 