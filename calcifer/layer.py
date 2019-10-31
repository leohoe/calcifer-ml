import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
	sig_x = sigmoid(x)
	return sig_x * (1 - sig_x)

def relu(x):
	return np.maximum(x, np.zeros(x.shape))

def relu_d(x):
	return np.full(x.shape,1) - (x <= 0)

def leaky_relu(x): 
	return x - (x <= 0) * x * 0.8

def leaky_relu_d(x):
	return 1 - (x <= 0) * 0.8

def linear(x):
	return x

def linear_d(x):
	return np.full(x.shape,1)
	
def softmax(x):
	safe_x = x - np.max(x)
	exp_x = np.exp(safe_x)
	result = exp_x / exp_x.sum(axis=1, keepdims=True)
	return result

def softmax_d(x):
	softmax_x = softmax(x)
	return np.diagflat(softmax_x.T) - (softmax_x.T @ softmax_x)

activationFunctions = {
	"sigmoid":		{"base": sigmoid,	"derivative": sigmoid_d},
	"relu":			{"base": relu,		"derivative": relu_d},
	"leaky_relu":	{"base": leaky_relu,"derivative": leaky_relu_d},
	"linear":		{"base": linear,	"derivative": linear_d},
	"softmax":		{"base": softmax,	"derivative": softmax_d},
}

class Layer:
	def __init__(self, sizeInput, sizeOutput, activation, learningRate, biasNeurons = 0, biasValue = 1):
		self.synapses		= 2 * np.random.rand(sizeInput + biasNeurons, sizeOutput) - 1
		self.activation		= activationFunctions[activation]["base"]
		self.activation_d	= activationFunctions[activation]["derivative"]
		self.learningRate	= learningRate
		self.biasNeurons	= biasNeurons
		self.biasValue		= biasValue
		self.usesCrossEntropy = (activation in ["softmax"])

	def feedForward(self, x):
		self.lastInput = x
		x = self.extendInputWithBias(x)

		self.lastValues = x @ self.synapses
		self.lastOutput = self.activation(self.clipValues(np.copy(self.lastValues)))

		return self.lastOutput

	def clipValues(self, x):
		return np.clip(x, -500, 500)

	def extendInputWithBias(self, x):
		if self.biasNeurons:
			biasMatrix = np.full((x.shape[0], self.biasNeurons), self.biasValue)
			x = np.hstack((x, biasMatrix))
		return x

	def removeBiasFromError(self, y):
		if self.biasNeurons:
			y = y[:, :-self.biasNeurons]
		return y

	def getLastOutput(self):
		return self.lastOutput

	def backpropagate(self, error):
		lastInputWithBias = self.extendInputWithBias(self.lastInput)
		delta = error * self.activation_d(self.clipValues(np.copy(self.lastValues)))
		adjustment = lastInputWithBias.T @ delta
		self.synapses += adjustment * self.learningRate

		previousError = delta @ self.synapses.T
		previousError = self.removeBiasFromError(previousError)

		return previousError
	
	def backpropagateCrossEntropy(self, target):
		lastInputWithBias = self.extendInputWithBias(self.lastInput)
		delta = -(self.lastOutput - target)
		adjustment = lastInputWithBias.T @ delta
		self.synapses += adjustment * self.learningRate

		previousError = delta @ self.synapses.T
		previousError = self.removeBiasFromError(previousError)

		return previousError

	def printSynapses(self):
		print(self.synapses)

	def checkUsesCrossEntropy(self):
		return self.usesCrossEntropy