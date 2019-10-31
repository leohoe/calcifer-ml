from mlxtend.data import loadlocal_mnist
import numpy as np 
import calcifer.neural_network as nn
from matplotlib import pyplot as plt
from matplotlib import gridspec
from PIL import Image
import os

ITERATION_COUNT = 20000
TEST_INTERVAL = 1000
TESTED_STARTING_ITERATIONS = 0
BATCH_SIZE = 128
USE_FASHION_MNIST = False

SAVE_WRONG_GUESSES = False
SAVE_WRONG_GUESSES_PATH = "./mnist_mistakes/"

np.random.seed(1)

def digitsToOneHot(x):
	output = []
	for label in x:
		output.append([1 if i == label else 0 for i in range(10)])
	return np.array(output)

def oneHotToDigits(y):
	return y.argmax(axis=1)

def crossEntropyLoss(raw_x, raw_y, epsilon=1e-8):
	x = np.array(raw_x).clip(epsilon, 1-epsilon)
	y = np.array(raw_y)
	return -np.sum(y * np.log(x)) / x.shape[0]

def correctGuessRatio(x,y):
	assert len(x) == len(y)
	correctGuesses = 0
	for i in range(len(x)):
		correctGuesses += (x[i] == y[i])
	return correctGuesses/len(x)

def initGraph():
	figure = plt.figure(tight_layout=True)
	figure.set_size_inches(10, 5)
	gridSpec = gridspec.GridSpec(2,2)
	axes = [
		figure.add_subplot(gridSpec[:,0]),
		figure.add_subplot(gridSpec[0,1]),
		figure.add_subplot(gridSpec[1,1]),
	]
	return figure, axes

def renderHistogram(axis, y1, y2):
	axis.cla()
	hist, xbins, ybins, im = axis.hist2d(result_labels, test_y_raw, cmap="plasma", range=np.array([[0,10], [0,10]]))
	axis.set_xticks([])
	axis.set_yticks([])
	axis.axis("square")
	axis.set_xlabel("Correct Label")
	axis.set_ylabel("Network Guess")

	for i in range(len(ybins)-1):
		for j in range(len(xbins)-1):
			axis.text(xbins[j]+0.5, ybins[i]+0.5, int(hist[j,i]), color="w", ha="center", va="center", fontweight="bold")

def renderErrorGraph(axis, error_history):
	axis.cla()
	axis.plot(error_history[0], error_history[1])
	axis.set_ylim(bottom=0)
	axis.set_xlabel("Iteration")
	axis.set_ylabel("Cross-Entropy Loss")

def renderAccuracyGraph(axis, accuracy_history):
	axis.cla()
	axis.plot(accuracy_history[0], accuracy_history[1])
	axis.set_ylim(top=1, bottom=0)
	axis.set_xlabel("Iteration")
	axis.set_ylabel("Test Set Accuracy")

train_x_raw, train_y_raw = loadlocal_mnist(
	"../" + ("fashion-" if USE_FASHION_MNIST else "") + "mnist/train-images-idx3-ubyte", 
	"../" + ("fashion-" if USE_FASHION_MNIST else "") + "mnist/train-labels-idx1-ubyte"
)

test_x_raw, test_y_raw = loadlocal_mnist(
	"../mnist/t10k-images-idx3-ubyte", 
	"../mnist/t10k-labels-idx1-ubyte"
)

train_x = np.array(train_x_raw / 255)
train_y = digitsToOneHot(train_y_raw)

test_x = np.array(test_x_raw / 255)
test_y = digitsToOneHot(test_y_raw)

batch_count = np.ceil(train_x.shape[0] / BATCH_SIZE)
batches_x = np.array_split(train_x, batch_count)
batches_y = np.array_split(train_y, batch_count)
batches_labels = np.array_split(train_y_raw, batch_count)
print("Generated",batch_count,"batches...")

neuralNet = nn.NeuralNetwork(784)
neuralNet.addLayer(64, "sigmoid")
neuralNet.addLayer(10, "softmax")

figure, axes = initGraph()
plt.pause(1)

error_history		= [[],[]]
accuracy_history	= [[],[]]

for i in range(1, ITERATION_COUNT+1):
	batch_number = int(i%batch_count)
	
	batch_x = batches_x[batch_number]
	batch_y = batches_y[batch_number]
	
	result = neuralNet.feedForward(batch_x)
	neuralNet.backpropagate(batch_y)

	if i in range(TESTED_STARTING_ITERATIONS) or i%TEST_INTERVAL == 0 or i == ITERATION_COUNT or np.log10(i)%1 == 0:
		test_result = neuralNet.feedForward(test_x)
		result_labels = oneHotToDigits(test_result)
		test_error = crossEntropyLoss(test_result, test_y)
		test_accuracy = correctGuessRatio(result_labels, test_y_raw)
		
		error_history[0].append(i)
		error_history[1].append(test_error)
		print(i, test_error, "\t|", test_accuracy*100, "% Correct")

		accuracy_history[0].append(i)
		accuracy_history[1].append(test_accuracy)

		renderHistogram(axes[0], result_labels, test_y_raw)
		renderErrorGraph(axes[1], error_history)
		renderAccuracyGraph(axes[2], accuracy_history)
		plt.tight_layout()
		plt.pause(0.001)

		if i == ITERATION_COUNT and SAVE_WRONG_GUESSES:
			wrong_guesses = [(j, test_x_raw[j], test_y_raw[j], result_labels[j]) for j in range(len(test_x)) if test_y_raw[j] != result_labels[j]]
			print("Saving",len(wrong_guesses), "wrong guesses...")
			
			for (imageID, imageData, imageLabel, guessedLabel) in wrong_guesses:
				imageData = np.array(imageData).reshape(28,28)
				image = Image.fromarray(imageData, mode="L")
				
				path = SAVE_WRONG_GUESSES_PATH + str(imageLabel) + "/"
				filename = str(imageID) + " - " + str(guessedLabel) + ".png"
				
				try:
					os.makedirs(path)
				except FileExistsError:
					pass
				
				image.save(path + filename)


plt.show()