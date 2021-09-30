import datetime
import os
import random
import time
import torch

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss.
	'''
	elapsed_rounded = int(round((elapsed)))
	return str(datetime.timedelta(seconds=elapsed_rounded))

def softmax(x):
	'''
	Takes logits and return probs.
	'''
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

class DatasetMLM(torch.utils.data.Dataset):
	def __init__(self, encodings):
		self.encodings = encodings
	def __getitem__(self, idx):
		return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
	def __len__(self):
		return len(self.encodings.input_ids)

class FastTransformer:
	def __init__(self, pretrained_path, num_labels, do_lower_case, batch_size, max_length, device, output_dir):
		self.model = AutoModelForSequenceClassification.from_pretrained(
			pretrained_path,              # Path to the pretrained model or name of huggingface model.
			num_labels = num_labels,      # Number of output labels (2 for binary classification).
			output_attentions = False,
			output_hidden_states = False,
			return_dict=False             # False returns logits for softmax
			)
		self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(
			pretrained_path,
			do_lower_case=do_lower_case
			)
		self.pretrained_path = pretrained_path
		self.num_labels = num_labels
		self.batch_size = batch_size
		self.max_length = max_length
		self.device = device
		self.output_dir = output_dir

	def set_optimizer(self, lr: float=2e-5, eps: float=1e-8) -> None:
		'''
		Creates the optimizer for training the transformer.

		Parameters
		__________

		lr: float (optional, defaults to 2e-5)
			The learning rate to use.
		eps: float (optional, defaults to 1e-8)
			Adam’s epsilon for numerical stability.

		Returns
		__________
		
		None

		'''
		# AdamW comes in the transformers module (vs Pytorch's native alternative)
		self.optimizer = AdamW(self.model.parameters(),
			lr = lr,
			eps = eps
			)

	def tokenize(self, X: list, labels: list) -> tuple:
		'''
		Tokenizes a list or array of documents.

		Parameters
		__________

		X: list
			The list of documents to tokenize.
		labels: list
			List of labels, parallel to documents.

		Returns
		__________
		
		input_ids: tensor
			Tensor of tokenized inputs.
		attention_masks: tensor
			Tensor identifying padded tokens.
		labels: tensor
			Tensor with the label associated to each document.
		'''

		tokenizer = self.tokenizer

		# Tokenizes texts and maps tokens to their IDs
		input_ids = []
		attention_masks = []

		# For each document...
		for text in X:
			# Things 'encode_plus' does:
			#   (1) Tokenizes documents.
			#   (2) Appends the '[CLS]' token at the beggining.
			#   (3) Appends the '[SEP]' token at the end.
			#   (4) Maps tokens to their IDs.
			#   (5) Adds padding or truncates documents according to the max_length.
			#   (6) Creates "attention masks" for the [PAD] tokens (padding).
			encoded_dict = tokenizer.encode_plus(
								text,
								add_special_tokens = True,    # '[CLS]' and '[SEP]'
								max_length = self.max_length,
								pad_to_max_length = True,
								return_attention_mask = True,
								return_tensors = 'pt',        # Returns PyTorch tensors
						  )

			# Adds the encoded text to the list of inputs for the model
			input_ids.append(encoded_dict['input_ids'])
			
			# Does the same with the attention masks
			attention_masks.append(encoded_dict['attention_mask'])

		# Converts the above lists to tensors
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		labels = torch.tensor(labels)

		return input_ids, attention_masks, labels

	def transform(self, X: list, y: list=[], sampler: str='random') -> DataLoader:
		'''
		Takes input documents and, optionally, labels and returns
		dataloaders with batches for training or making predictions.

		Parameters
		__________

		X: list
			The list of input documents.
		y: list
			List of labels, parallel to documents.
		sampler: str
			String indicating whether to create batches at random or sequentially

		Returns
		__________
		
		dataloader: DataLoader
			DataLoader with batches created.

		'''

		# If no labels are provided (testing model), create dummy labels that will be ignored
		if len(y) == 0:
			y = [0]*len(X)

		input_ids, attention_masks, labels = self.tokenize(X, y)

		dataset = TensorDataset(input_ids, attention_masks, labels)

		# Random sampler for training and sequential for testing
		if sampler == 'random':
			sampler = RandomSampler(dataset)
		elif sampler == 'sequential':
			sampler = SequentialSampler(dataset)

		dataloader = DataLoader(
			dataset,
			sampler = sampler,
			batch_size = self.batch_size
			)

		return dataloader

	def pretrain_mlm(self, input_text: list, epochs: int=1,
					 mlm_probability: float=0.15, output_dir: str='pretrained_mlm',
					 update_classifier_pretrained_model: bool=True) -> None:
		'''
		Takes input documents and, optionally, labels and returns
		dataloaders with batches for training or making predictions.

		Parameters
		__________

		input_text: list
			Documents to be used for training with masked language.

		epochs: int (optional, defaults to 1)
			Times the entire dataset will be used for training.

		mlm_probability: float (optional, defaults to 0.15)
			Probability for each token to get masked.

		output_dir: str
			Path to the folder where the model should get saved at.

		update_classifier_pretrained_model: bool
			If True, the pretrained model defined when the class was first instanced
			will be replaced with the model fine-tuned on masked language.

		Returns
		__________
		
		None

		'''

		# Passes the inputs through the tokenizer
		inputs = self.tokenizer(
			input_text,
			return_tensors='pt',
			max_length=self.max_length,
			truncation=True,
			padding='max_length'
			)

		# We're going to be masking tokens at random. So we...
		# 1) create a copy of the original tokens as labels
		# 2) assign random float in the interval (0, 1) for each token
		# 3) if the random float is < mlm_probability and it isn't a [CLS], [SEP] or [PAD] token we mask it
		inputs['labels'] = inputs.input_ids.detach().clone()
		rand = torch.rand(inputs.input_ids.shape)
		mask_arr = (rand < mlm_probability) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
		selection = []
		for i in range(inputs.input_ids.shape[0]):
			selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
			inputs.input_ids[i, selection[i]] = 103

		# Training dataloader
		dataset = DatasetMLM(inputs)
		loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

		# Instance of the model + send it to cuda if available
		model = AutoModelForMaskedLM.from_pretrained(self.pretrained_path)
		model.to(self.device)

		if self.device == 'cuda':
			torch.cuda.empty_cache()

		# Begin training
		model.train()
		optim = AdamW(model.parameters(), lr=5e-5)

		for epoch in range(epochs):
			# Setup loop with TQDM and dataloader
			loop = tqdm(loader, leave=True)
			for batch in loop:
				# Initialize calculated gradients (from prev step)
				optim.zero_grad()
				# Pull all tensor batches required for training
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)
				labels = batch['labels'].to(self.device)
				# Process
				outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
				# Extract loss
				loss = outputs.loss
				# Calculate loss for every parameter that needs grad update
				loss.backward()
				# Update parameters
				optim.step()
				# Print relevant info to progress bar
				loop.set_description(f'Epoch {epoch}')
				loop.set_postfix(loss=loss.item())

		# Create the output dir if it doesn't already exist
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# Save the model
		# It can leater be used calling the method `from_pretrained()`
		model_to_save = model.module if hasattr(model, 'module') else model
		model_to_save.save_pretrained(output_dir)
		self.tokenizer.save_pretrained(output_dir)

		if update_classifier_pretrained_model:
			# Update the classifier model with the new language model
			self.model = AutoModelForSequenceClassification.from_pretrained(
				output_dir,
				num_labels = self.num_labels,
				output_attentions = False,
				output_hidden_states = False,
				return_dict=False
				)


	def train_classifier(self, dataloader: DataLoader, epochs: int=1, random_state: int=42) -> None:
		'''
		Takes input documents and, optionally, labels and returns
		dataloaders with batches for training or making predictions.

		Parameters
		__________

		dataloader: DataLoader
			DataLoader with batches of tokenized documents for training.

		epochs: int (optional, defaults to 1)
			Times the entire dataset will be used for training.


		Returns
		__________
		
		None

		'''

		self.model.to(self.device)

		params = list(self.model.named_parameters())

		# Number of total steps is [number of batches] x [number of epochs]
		# (This is not the same as the nbr of training samples)
		total_steps = len(dataloader) * epochs

		# Create the learning rate scheduler
		scheduler = get_linear_schedule_with_warmup(
			self.optimizer,
			num_warmup_steps = 0,
			num_training_steps = total_steps
			)

		random.seed(random_state)
		np.random.seed(random_state)
		torch.manual_seed(random_state)
		if self.device == 'cuda':
			torch.cuda.empty_cache()
			torch.cuda.manual_seed_all(random_state)

		# We'll store training loss and training time here
		training_stats = []

		# Init time for calculating how long the entire training process takes
		total_t0 = time.time()

		# For each epoch...
		for epoch_i in range(epochs):
			
			# ========================================
			#               Training
			# ========================================

			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
			print('Training...')

			# For calculating the time taken by the current epoch
			t0 = time.time()

			# Reset the total loss for this epoch
			total_train_loss = 0

			# We put the model in training mode
			self.model.train()

			# Iterating over training batches...
			for step, batch in enumerate(dataloader):

				# Print process every 40 batches
				if step % 40 == 0 and not step == 0:
					elapsed = format_time(time.time() - t0)
					print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

				# Get tensors and send them to device
				b_input_ids = batch[0].to(self.device)
				b_input_mask = batch[1].to(self.device)
				b_labels = batch[2].to(self.device)

				# Clear out the gradients (by default they accumulate)
				self.model.zero_grad()        

				# Forward pass
				loss, logits = self.model(
					b_input_ids,
					attention_mask=b_input_mask,
					labels=b_labels
					)

				# Accumulate the training loss over all of the batches
				total_train_loss += loss.item()
				# Perform a backward pass to calculate the gradients
				loss.backward()
				# Clip the norm of the gradients to 1.0
				# This is to help prevent the "exploding gradients" problem
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				# Update parameters and take a step using the computed gradient
				self.optimizer.step()
				# Update the learning rate
				scheduler.step()

			# Calculate the average loss over all of the batches
			avg_train_loss = total_train_loss / len(dataloader)            
			
			# Measure how long this epoch took
			training_time = format_time(time.time() - t0)

			print("")
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
			print("  Training epoch took: {:}".format(training_time))

		print("")
		print("Training complete!")

		print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

		# Create the output dir if it doesn't already exist
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		print("Saving model to %s" % self.output_dir)

		# Save the model
		# It can leater be used calling the method `from_pretrained()`
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
		model_to_save.save_pretrained(self.output_dir)
		self.tokenizer.save_pretrained(self.output_dir)

	def predict(self, dataloader):
		# Put model in evaluation mode
		self.model.eval()

		# Tracking variable
		predictions = []

		# Predict.
		for batch in dataloader:
			# Add batch to GPU, if available
			batch = tuple(t.to(self.device) for t in batch)
			
			# Unpack the inputs from our dataloader
			b_input_ids, b_input_mask, b_labels = batch
			
			# Telling the model not to compute or store gradients, saving memory and 
			# speeding up prediction
			with torch.no_grad():
				# Forward pass, calculate logits (predictions prior to softmax)
				outputs = self.model(
					b_input_ids,
					attention_mask=b_input_mask
					)

			logits = outputs[0]

			# Move logits to CPU, if on GPU
			logits = logits.detach().cpu().numpy()

			# Store predictions
			for pred in logits:
				predictions.append(tuple(softmax(pred)))

		return predictions