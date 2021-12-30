import os
import pickle

import numpy as np

from PIL import Image
from glob import glob
from tqdm import tqdm

from matplotlib import pyplot as plt



class Dataset:
	translate = {
		"cane": "dog",
		"cavallo": "horse",
		"elefante": "elephant",
		"farfalla": "butterfly",
		"gallina": "chicken",
		"gatto": "cat",
		"mucca": "cow",
		"pecora": "sheep",
		"ragno": "spider",
		"scoiattolo": "squirrel",
		"dog": "cane",
		"horse": "cavallo",
		"elephant" : "elefante",
		"butterfly": "farfalla",
		"chicken": "gallina",
		"cat": "gatto",
		"cow": "mucca",
		"sheep": "pecora",
		"spider": "ragno",
		"squirrel": "scoiattolo"
	}

	def __init__(self, data_parent='data/'):
		super().__init__()

		self.data_path = os.path.join(data_parent, 'raw-img')

		self.org_names = list(self.translate.keys())[:10]
		self.eng_names = list(self.translate.keys())[10:]

		self.data_paths = []
		for i in range(len(self.org_names)):
			img_paths = glob(
				os.path.join(self.data_path, self.org_names[i], '*')
			)

			self.data_paths.extend(list(zip(img_paths, [i] * len(img_paths))))

		self.mean = 0.459
		self.var = 0.269

	def __len__(self):
		return len(self.data_paths)

	def __getitem__(self, index):
		img_path, label = self.data_paths[index]
		img = Image.open(img_path).convert('L').resize((64, 64))
		img_arr = np.asarray(img)
		img_arr = img_arr / 255
		img_arr = (img_arr - self.mean) / self.var
		img_arr = img_arr.flatten()

		return img_arr, label

	def getlabel(self, index):
		return self.data_paths[index][1]


def fully_connected(X, params):
	W, b = params

	out = (X @ W) + b
	dx = W
	dw = X
	db = np.ones_like(out)  # Not used

	return {'out': out, 'dx': dx, 'dw': dw, 'db': db}

def LRelu(X, a):
	out = np.where(X<0, X*a, X)

	# Derivative for Gradient Descent
	dx = np.where(X<0, a, 1)

	return {'out': out, 'dx': dx}

def softmax_nll(X, Y):
	# Extract max for numerical stability.
	x_max = np.max(X, axis=1, keepdims=True)

	# Unstable
	# e_x = np.exp(X - x_max)
	# probs = e_x / np.sum(e_x, axis=1, keepdims=True)

	log_e_x = np.log(np.sum(np.exp(X - x_max), axis=1, keepdims=True))
	log_probs = X - x_max - log_e_x

	# Compute loss
	N = X.shape[0]
	out = -np.sum(log_probs[np.arange(N), Y]) / N

	# Derivative for Gradient Descent
	dx = np.exp(log_probs)
	dx[np.arange(N), Y] = dx[np.arange(N), Y] - 1
	dx = dx/N

	return {'out': out, 'dx': dx}

def softmax_nll_eval(X, Y):
	# Extract mean for numerical stability.
	x_max = np.max(X, axis=1, keepdims=True)

	# Unstable
	# e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
	# probs = e_x / np.sum(e_x,axis=1, keepdims=True)

	log_e_x = np.log(np.sum(np.exp(X - x_max), axis=1, keepdims=True))
	log_probs = X - x_max - log_e_x

	return {'out': np.argmax(log_probs, axis=1)}

def test(network, dataset, test_set, batch_size, epoch):
	iter_count = len(test_set) // batch_size

	data_index = 0
	epoch_tqdm = tqdm(
		range(iter_count), total=iter_count, leave=True, desc=f"Test {epoch}"
	)

	targets = []
	predictions = []
	for iter in epoch_tqdm:
		X = []
		Y = []
		for _ in range(batch_size):
			x, y = dataset[train_set[data_index]]
			X.append(x)
			Y.append(y)
			data_index += 1
		X = np.asarray(X)
		Y = np.asarray(Y)

		output_of_previous_layer = X

		for f in network:
			if f == softmax_nll_eval:
				y = f(output_of_previous_layer, None)
			else:
				y = f[0](output_of_previous_layer, f[1])
			output_of_previous_layer = y['out']

		predictions.extend(y['out'])
		targets.extend(Y)

	return np.asarray(targets), np.asarray(predictions)

def train_one_epoch(network, dataset, train_set, lr, batch_size, epoch):
	iter_count = len(train_set) // batch_size

	loss_vals = []

	data_index = 0
	epoch_tqdm = tqdm(
		range(iter_count), total=iter_count, leave=True, desc=f"Train {epoch}:"
	)
	for iter in epoch_tqdm:
		X = []
		Y = []
		for _ in range(batch_size):
			x, y = dataset[train_set[data_index]]
			X.append(x)
			Y.append(y)
			data_index += 1
		X = np.asarray(X)
		Y = np.asarray(Y)

		output_of_previous_layer = X

		out_list = []
		for f in network:
			if f == softmax_nll:
				y = f(output_of_previous_layer, Y)
			else:
				y = f[0](output_of_previous_layer, f[1])
			out_list.append(y)
			output_of_previous_layer = y['out']

		loss_vals.append(y['out'])

		dout = out_list[-1]['dx']
		for i in range(len(network)-2, -1, -1):
			if network[i][0] == fully_connected:
				network[i][1][0] = (
					network[i][1][0] - lr * out_list[i]['dw'].T @ dout
				)
				network[i][1][1] = (
					network[i][1][1] - lr * np.sum(dout, axis=0, keepdims=True)
				)

				dout = dout @ out_list[i]['dx'].T

			elif network[i][0] == LRelu:
				dout = dout * out_list[i]['dx']

	return network, loss_vals



if __name__ == "__main__":
	file = open('results.txt', 'w')
	global best_of_best_network
	np.random.seed(123)
	batch_size = 16
	dataset = Dataset()

	test_set_size = len(dataset)//10
	train_set_size = len(dataset)-test_set_size

	indices = np.arange(len(dataset))
	indices = np.random.permutation(indices)

	train_set = indices[:train_set_size]
	test_set = indices[train_set_size:]

	# Discard last batch if not compatible with batch size
	test_set_size = (test_set_size // batch_size) * batch_size
	train_set_size = (train_set_size // batch_size) * batch_size
	train_set = indices[:train_set_size]
	test_set = indices[:test_set_size]

	model_name = 'mlnn.pth'
	is_train = False

	# Number of units in network
	neurons_list = [[4096, 10], [4096, 1024, 10], [4096, 1024, 256, 10]]
	batch_sizes = [16, 32, 64, 128]
	lrs = [0.005, 0.01, 0.02]
	#lr = 0.01

	best_of_best_acc = -100
	for neurons in neurons_list: # 3
		for batch_size in batch_sizes: # 4
			for lr in lrs: # 3
				# Create model
				xav_lim = 1 / np.sqrt(neurons[0])
				network = []
				for i in range(1, len(neurons)):
					network.append([
						fully_connected,
						[  # Xavier initialization: [-1/sqrt_inp_neur, 1/sqrt_inp_neur]
							np.random.rand(
								neurons[i-1], neurons[i]
							) * 2 * xav_lim - xav_lim,  # Weights
							np.zeros((1, neurons[i]))  # Biases
						]
					])

					if i != len(neurons) - 1:
						network.append([LRelu, 0.1])
					else:
						network.append(softmax_nll)

				# Training loop
				best_network = []
				best_acc = -100
				full_loss_vals = []
				for epoch in range(1, 10+1):
					file.write("\n------------------------------------------------------------\n")
					file.write("Neurons: {}, Batch Size: {}, Learning Rate: {}, Epoch: {}\n".format(neurons, batch_size, lr, epoch))
					# Train
					network[-1] = softmax_nll
					network, loss_vals = train_one_epoch(
						network, dataset, train_set, lr, batch_size, epoch
					)

					file.write(f"Loss Value: {sum(loss_vals)/len(loss_vals):.2f}%\n")

					# LR scheduler
					#lr *= 0.9

					full_loss_vals.extend(loss_vals)

					# Eval
					network[-1] = softmax_nll_eval
					targets, predictions = test(
						network, dataset, test_set, batch_size, epoch
					)

					train_acc = np.mean(targets == predictions) * 100

					file.write(f"Train Accuracy: {train_acc:.2f}%\n")

					if train_acc >= best_acc:
						best_network = network
						#pickle.dump(network, open(model_name, 'wb'))
						best_acc = train_acc
					if train_acc >= best_of_best_acc:
						best_of_best_network = network
						best_of_best_acc = train_acc

				#plt.plot(full_loss_vals)
				#plt.savefig("part1_train_loss_vals.png")

				#model_path = "models/mlnn_2.pth"
				#network = pickle.load(open(model_path, 'rb'))

				# Eval
				best_network[-1] = softmax_nll_eval
				targets, predictions = test(
					best_network, dataset, test_set, batch_size, "Run"
				)

				conf_mat = np.zeros((10, 10))
				for t, p in zip(targets, predictions):
					conf_mat[t, p] += 1

				test_acc = np.mean(targets == predictions) * 100
				file.write(f"Accuracy: {test_acc:.2f}%\n")

				file.write("Confusion matrix:\n")
				content = str(conf_mat)
				file.write(content)
				print(conf_mat)

				file.write("\n*******************************************\n")
	file.close()

best_of_best_network = []
