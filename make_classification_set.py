import numpy as np
import pickle
from matplotlib import pyplot as plt

CLASS_COUNT		= 3
ITEMS_PER_CLASS	= 700
RENDER_GRAPH	= True

np.random.seed(1)

group_0 = np.random.randn(ITEMS_PER_CLASS, 2) + np.array([0, -3])
group_1 = np.random.randn(ITEMS_PER_CLASS, 2) + np.array([3, 3])
group_2 = np.random.randn(ITEMS_PER_CLASS, 2) + np.array([-3, 3])

feature_set	= np.vstack([group_0, group_1, group_2])
labels		= np.array([0]*ITEMS_PER_CLASS + [1]*ITEMS_PER_CLASS + [2]*ITEMS_PER_CLASS)

one_hot_labels	= np.zeros((ITEMS_PER_CLASS * CLASS_COUNT, CLASS_COUNT))
for i in range(ITEMS_PER_CLASS * CLASS_COUNT):
	one_hot_labels[i, labels[i]] = 1

training_data = {}
training_data["input"] = feature_set
training_data["output"] = one_hot_labels
training_data["labels"] = labels

with open("example_set_classification.pickle", "wb") as outfile:
	pickle.dump(training_data, outfile)

if RENDER_GRAPH:
	plt.scatter(training_data["input"][:,0], training_data["input"][:,1], c=training_data["output"], alpha=0.5)
	plt.show()