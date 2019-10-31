import numpy as np
from calcifer.layer import Layer

class NeuralNetwork:
	def __init__(self, 
			inputLayerSize, 
			learningRate		= 0.01, 
			biasNeuronsPerLayer	= 1,
			):
		self.layers = []
		self.lastLayerSize = inputLayerSize
		self.learningRate = learningRate
		self.biasNeuronsPerLayer = biasNeuronsPerLayer

	def addLayer(self, outputLayerSize, activation):
		newLayer = Layer(
			sizeInput	= self.lastLayerSize, 
			sizeOutput	= outputLayerSize, 
			activation	= activation, 
			learningRate = self.learningRate,
			biasNeurons = self.biasNeuronsPerLayer)
		self.layers.append(newLayer)
		self.lastLayerSize = outputLayerSize

	def feedForward(self, x):
		lastNodes = np.array(x, ndmin=2)
		for layer in self.layers:
			lastNodes = layer.feedForward(lastNodes)
		self.lastOutput = lastNodes
		return self.lastOutput

	def getLastOutput(self):
		return self.lastOutput
	
	def	printSynapses(self):
		for layer in self.layers:
			layer.printSynapses()

	def backpropagate(self, target):
		outputLayer = self.layers[-1]
		if outputLayer.checkUsesCrossEntropy():
			lastError = outputLayer.backpropagateCrossEntropy(target)
		else:
			lastError = outputLayer.backpropagate(target - self.lastOutput)

		for layer in reversed(self.layers[:-1]):
			lastError = layer.backpropagate(lastError)
		pass