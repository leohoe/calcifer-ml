import calcifer.neural_network as nn
import calcifer.layer
import numpy as np
from matplotlib import pyplot as plt
import pickle

ITERATION_COUNT	= 100
BATCH_SIZE		= 60
TEST_RATIO		= 0.1

np.random.seed(1)

def crossEntropyLoss(raw_x, raw_y, epsilon=1e-8):
	x = np.array(raw_x).clip(epsilon, 1-epsilon)
	y = np.array(raw_y)
	return -np.sum(y * np.log(x)) / x.shape[0]

def shuffleTrainingData(x, y):
	joined = np.hstack((x,y))
	np.random.shuffle(joined)
	separated = np.hsplit(joined, [x.shape[1]])
	return (separated[0], separated[1])

def normalizeInputs(x):
	return x / np.max(x, axis=0)

def generateTrainingBatches(x, y):
	x, y = shuffleTrainingData(x, y)
	x_batches = np.array_split(x, batch_count)
	y_batches = np.array_split(y, batch_count)
	return x_batches, y_batches

def initGraph():
	figure, axes = plt.subplots(2, 2)
	figure.set_size_inches(7, 7)
	return figure, axes

def renderErrorGraph(axis, errorValues):
	axis.cla()

	axis.set_title("Training Error")
	axis.set_xlabel("Iteration")
	axis.set_ylabel("Mean Error")
	axis.set_ylim(bottom=0, top=1)
	axis.margins(y=0)
	axis.plot(errorValues)

def renderClassificationGraph(axis, x, y):
	axis.cla()

	axis.set_title("Test Classification")
	axis.axis("equal")
	axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
	axis.scatter(x[:,0], x[:,1], c=y)

grid_set = np.array([[x/10,y/10] for x in range(-10,11) for y in range(-10,11)])

x = y = None
with open("example_set_classification.pickle", "rb") as infile:
	data = pickle.load(infile)
	x = data["input"]
	y = data["output"]
	assert x.shape[0] == y.shape[0]

batch_count = np.round(x.shape[0] / BATCH_SIZE)
test_data_split_index = int(np.round(x.shape[0] * (1-TEST_RATIO)))

x = normalizeInputs(x)
x, y = shuffleTrainingData(x, y)

split_x = np.vsplit(x, [test_data_split_index])
split_y = np.vsplit(y, [test_data_split_index])
training_x = split_x[0]
training_y = split_y[0]
test_x = split_x[1]
test_y = split_y[1]

neuralNet = nn.NeuralNetwork(x.shape[1])
neuralNet.addLayer(6, "leaky_relu")
neuralNet.addLayer(y.shape[1], "softmax")

errorValues = []
figure, axes = initGraph()
plt.pause(1)

for i in range(1, ITERATION_COUNT+1):
	test_result = neuralNet.feedForward(test_x)
	errorValues.append(crossEntropyLoss(test_result, test_y))

	if i < 20 or i % 100 == 0 or np.log10(i)%1 == 0:
		print("Iteration ", i, ":\t", errorValues[-1])
		renderClassificationGraph(axes[0,0], test_x, test_result)
		renderClassificationGraph(axes[1,0], grid_set, neuralNet.feedForward(grid_set))
		renderErrorGraph(axes[0,1], errorValues)
		plt.tight_layout()
		plt.pause(0.001)
	
	x_batches, y_batches = generateTrainingBatches(training_x,training_y)
	for j in range(len(x_batches)):
		result = neuralNet.feedForward(x_batches[j])
		neuralNet.backpropagate(y_batches[j])

test_result = neuralNet.feedForward(test_x)
print(crossEntropyLoss(test_result, test_y))

renderClassificationGraph(axes[0,0], test_x, test_result)
renderClassificationGraph(axes[1,0], grid_set, neuralNet.feedForward(grid_set))
renderErrorGraph(axes[0,1], errorValues)
plt.tight_layout()
plt.show()