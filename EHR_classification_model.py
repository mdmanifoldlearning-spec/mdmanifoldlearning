import math
import gensim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class EHRClassifier(nn.ModuleList):

	def __init__(self, params, word2vec):
		super(EHRClassifier, self).__init__()

		# Parameters regarding text preprocessing
		self.embedding_size = params['word_embedding_size']
		self.tabular_feature_size = params['tabular_feature_size']
		self.rnn_input_dim = params['rnn_input_dim']
		self.rnn_hidden_dim = params['rnn_hidden_dim']
		self.rnn_num_layers = params['rnn_num_layers']
		self.fcn_hidden_dim = params['fcn_hidden_dim']
		# Dropout definition
		self.dropout1 = nn.Dropout(0.2)
		self.dropout2 = nn.Dropout(0.2)
		
		# CNN parameters definition
		# Kernel sizes
		self.kernel_1 = 2
		self.kernel_2 = 4
		self.kernel_3 = 8
		self.kernel_4 = 16
		
		# Output size for each convolution
		self.out_size = params['cnn_out_size']
		
		# Embedding layer definition
		weights = torch.FloatTensor(word2vec.vectors)
		self.embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
		
		# Convolution layers definition
		self.conv_1 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_1, 1)
		self.conv_2 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_2, 1)
		self.conv_3 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_3, 1)
		self.conv_4 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_4, 1)
		
		# rnn layers definition
		self.rnn = nn.LSTM(input_size=self.rnn_input_dim, hidden_size=self.rnn_hidden_dim, 
			num_layers=self.rnn_num_layers, batch_first=True, dropout=0.3)

		# Fully connected layer definition
		# linear_in_dim = self.out_size*4+self.rnn_hidden_dim+self.tabular_feature_size
		linear_in_dim = self.out_size*4
		self.fc1 = nn.Linear(linear_in_dim, self.fcn_hidden_dim)
		self.fc2 = nn.Linear(self.fcn_hidden_dim, 1)		
		
	def forward(self, text_input, ts_input, tabular_input):

		# Sequence of tokes is filterd through an embedding layer
		text_input = self.embedding(text_input)
		text_input = torch.transpose(text_input, -2, -1)
		
		# Convolution layers on text input
		text_x1 = self.conv_1(text_input)
		text_x1, _ = torch.max(text_x1, dim=2)
		text_x2 = self.conv_2(text_input)
		text_x2, _ = torch.max(text_x2, dim=2)
		text_x3 = self.conv_3(text_input)
		text_x3, _ = torch.max(text_x3, dim=2)
		text_x4 = self.conv_4(text_input)
		text_x4, _ = torch.max(text_x4, dim=2)
		# rnn on ts input
		# rnn_out, _ = self.rnn(ts_input)
		# last_rnn_out = rnn_out[:, -1, :]
		# union = torch.cat((text_x1, text_x2, text_x3, text_x4, last_rnn_out, tabular_input), 1)

		union = torch.cat((text_x1, text_x2, text_x3, text_x4), 1)

		out = self.fc1(union)
		out = self.dropout1(out)
		out = self.fc2(out)
		out = self.dropout2(out)
		# Activation function is applied
		out = torch.sigmoid(out)
		
		return out.squeeze()

class DatasetMaper(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


class Run:
	@staticmethod
	def get_weighted_sampler(y):
		num1 = np.sum(y == 1)
		num0 = np.sum(y == 0)
		weight = {0: 1./num0*1000, 1: 1./num1*1000}
		samples_weight = np.array([weight[t] for t in y])
		samples_weight = torch.from_numpy(samples_weight)
		sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
		return sampler
	@staticmethod
	def train(model, data, params, md_manifold_y_pred_test=None):
		lr = params['lr']
		batch_size = params['batch_size']
		epochs = params['epochs']
		device = params['device']
		# textonly = params['textonly']

		# Initialize dataset maper
		train = DatasetMaper(data['x_train'], data['y_train'])
		test = DatasetMaper(data['x_test'], data['y_test'])
		
		# Initialize loaders
		loader_train = DataLoader(train, batch_size=batch_size, \
			sampler=Run.get_weighted_sampler(data['y_train']), drop_last=True)
		loader_test = DataLoader(test, batch_size=batch_size, shuffle=False)
		
		# Define optimizer
		optimizer = optim.Adam(model.parameters(), lr=lr)
		
		model = model.to(device=device)
		# Starts training phase
		for epoch in range(epochs):
			# Set model in training model
			model.train()
			train_predictions = []
			train_targets = []
			# Starts batch training
			for x_batch, y_batch in loader_train:
				y_batch = y_batch.type(torch.FloatTensor)
				y_batch = y_batch.to(device=device)
				if len(x_batch) > 1:
					x_batch[1] = x_batch[1].type(torch.FloatTensor)
					x_batch[2] = x_batch[2].type(torch.FloatTensor)
					x_batch = [item.to(device=device) for item in x_batch]
				else:
					x_batch = x_batch.to(device=device)
				# Feed the model
				if len(x_batch) > 1:
					y_pred = model(*x_batch)
				else:
					y_pred = model(x_batch)

				loss = F.binary_cross_entropy(y_pred, y_batch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				# Save predictions
				train_predictions += list(y_pred.detach().cpu().numpy())
				train_targets += list(y_batch.detach().cpu().numpy())
			
		# Evaluation phase
		test_predictions = Run.evaluation(model, loader_test, device)

		# Metrics calculation
		train_predictions01 = np.array(train_predictions) > 0.5
		train_accuracy = accuracy_score(train_targets, train_predictions01)
		train_auc = roc_auc_score(train_targets, train_predictions)

		test_predictions01 = np.array(test_predictions) > 0.5
		test_accuracy = accuracy_score(data['y_test'], test_predictions01)
		test_auc = roc_auc_score(data['y_test'], test_predictions)
		test_precision = precision_score(data['y_test'], test_predictions01)
		test_recall = recall_score(data['y_test'], test_predictions01)
		test_f1 = f1_score(data['y_test'], test_predictions01)
		
		if md_manifold_y_pred_test is not None:
			perf_df = Run.ensemble_perf(np.array(test_predictions), md_manifold_y_pred_test, data['y_test'])
		else:
			perf_df = pd.DataFrame()


		perf_df['train_accuracy'] = [train_accuracy]
		perf_df['train_auc'] = [train_auc]
		perf_df['ehr_test_accuracy'] = [test_accuracy]
		perf_df['ehr_test_auc'] = [test_auc]
		perf_df['ehr_test_precision'] = [test_precision]
		perf_df['ehr_test_recall'] = [test_recall]
		perf_df['ehr_test_f1'] = [test_f1]
		return perf_df

	@staticmethod
	def ensemble_perf(test_predictions1, test_predictions2, test_target):
		ensemble_accuracy = []
		ensemble_auc = []
		ensemble_precision = []
		ensemble_recall = []
		ensemble_f1 = []
		ensemble_a = []
		for a in np.arange(0,1,0.01):
			ensemble_predictions_avg = a*test_predictions1 + (1-a)*test_predictions2
			ensemble_predictions_avg01 = ensemble_predictions_avg > 0.5
			ensemble_accuracy.append(accuracy_score(test_target, ensemble_predictions_avg01))
			ensemble_auc.append(roc_auc_score(test_target, ensemble_predictions_avg))
			ensemble_precision.append(precision_score(test_target, ensemble_predictions_avg01))
			ensemble_recall.append(recall_score(test_target, ensemble_predictions_avg01))
			ensemble_f1.append(f1_score(test_target, ensemble_predictions_avg01))
			ensemble_a.append(a)
		perf_df = pd.DataFrame()
		perf_df['ensemble_accuracy'] = ensemble_accuracy
		perf_df['ensemble_auc'] = ensemble_auc
		perf_df['ensemble_precision'] = ensemble_precision
		perf_df['ensemble_recall'] = ensemble_recall
		perf_df['ensemble_f1'] = ensemble_f1
		perf_df['ensemble_a'] = ensemble_a
		return perf_df

	@staticmethod
	def evaluation(model, loader_test, device):
		
		# Set the model in evaluation mode
		model.eval()
		predictions = []
		
		# Starst evaluation phase
		with torch.no_grad():
			for x_batch, y_batch in loader_test:
				if len(x_batch) > 1:
					x_batch[1] = x_batch[1].type(torch.FloatTensor)
					x_batch[2] = x_batch[2].type(torch.FloatTensor)
					x_batch = [item.to(device=device) for item in x_batch]
					y_pred = model(*x_batch)
				else:
					x_batch = x_batch.to(device=device)
					y_pred = model(x_batch)
				predictions += list(y_pred.detach().cpu().numpy())
		return predictions
		

