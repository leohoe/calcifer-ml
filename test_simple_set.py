from example_set_simple import binary_to_one_hot as example
import calcifer.neural_network as nn
import numpy as np
import matplotlib.pyplot as plt

def calculateMeanError(errorArray):
	return np.mean(np.abs(errorArray))

np.set_printoptions(precision=4, suppress=True,linewidth=100)
np.random.seed(1)

x = np.array(example["input"])
y = np.array(example["output"])

neuralNet = nn.NeuralNetwork(
	inputLayerSize=x.shape[1], 
	learningRate=0.01, 
	biasNeuronsPerLayer=1)
neuralNet.addLayer(10,			"leaky_relu")
neuralNet.addLayer(y.shape[1],	"softmax")

errorValues = []

for i in range(10000):
	result = neuralNet.feedForward(x)
	neuralNet.backpropagate(y)

	errorValues.append(calculateMeanError(y - result))

	if i == 0 or np.log2(i) % 1 == 0:
		print(i)
		print("Error: " + str(calculateMeanError(y - result)))

print("Done! Final Error:")
print(y - result)
print("Mean: " + str(calculateMeanError(y - result)))

print("Result:")
print(result)

plt.title("Training Error")
plt.xlabel("Iteration")
plt.ylabel("Mean Error")
plt.ylim(bottom=0, top=1)
plt.margins(y=0)
plt.plot(errorValues)
plt.show()