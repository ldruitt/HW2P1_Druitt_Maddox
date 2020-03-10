# Print a bar chart with groups

import numpy as np
import matplotlib.pyplot as plt

# set height of bar
# length of these lists determine the number
# of groups (they must all be the same length)
bars1 = [10.8, 11.3, 98.5, 99.0, 87, 88, 12]
bars2 = [0, 0, 0, 0, 87, 87, 2]

# set width of bar. To work and supply some padding
# the number of groups times barWidth must be
# a little less than 1 (since the next group
# will start at 1, then 2, etc).

barWidth = 0.25
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='Accuracy')
plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='F1')

# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Keras1', 'Keras2', 'Keras3', 'Keras4', 'SVM1', 'SVM2', 'SVM3'])

# Create legend & Show graphic
plt.legend()
plt.show()
#plt.savefig("barChart.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf