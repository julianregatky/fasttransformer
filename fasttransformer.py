import datetime
import os
import random
import time
import torch

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

def format_time(elapsed):
	'''
	Toma timpo en segundos y los devuelve como string con formato hh:mm:ss
	'''
	elapsed_rounded = int(round((elapsed)))
	return str(datetime.timedelta(seconds=elapsed_rounded))

class FastTransformer:
	def __init__(self, model, num_labels, do_lower_case, batch_size, epochs, max_length, device, output_dir):
		self.model = AutoModelForSequenceClassification.from_pretrained(
			model,                        # El modelo BERT seleccioando.
			num_labels = num_labels,      # Nro de labels de salida (2 para clasif. binaria).
			output_attentions = False,    # Si queremos que devuelva los "attention weights".
			output_hidden_states = False, # Si queremos que devuelva los hidden states.
			return_dict=False             # Le pedimos que nos devuelva los logits para
										  # poder aplicarles softmax y obtener probs.
			)
		self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(
			model,
			do_lower_case=do_lower_case
			)
		self.batch_size = batch_size
		self.epochs = epochs              # Nro de épocas (veces que el modelo ve el dataset de training entero).
		self.max_length = max_length
		self.device = device
		self.output_dir = output_dir

	def set_optimizer(self, lr=2e-5, eps=1e-8):
		# Preparamos el optimizador.
		# Usamos AdamW del módulo transformers (vs uno nativo de Pytorch).
		self.optimizer = AdamW(self.model.parameters(),
			lr = lr,
			eps = eps
			)

	def tokenize(self, X, labels):

		tokenizer = self.tokenizer

		# Tokenizamos todos los textos y mapeamos los tokens a sus IDs
		input_ids = []
		attention_masks = []

		# Para cada texto...
		for text in X:
			# Cosas que hace 'encode_plus':
			#   (1) Tokenize los textos.
			#   (2) Pre-appendea el token '[CLS]' al principio.
			#   (3) Appendea el token '[SEP]' al final.
			#   (4) Mapea tokens a sus IDs.
			#   (5) Agrega paddings o trunca los textos de acuerdo al max_length.
			#   (6) Crea "attention masks" para los tokens [PAD] de padding.
			encoded_dict = tokenizer.encode_plus(
								text,                              # Texto a encodear.
								add_special_tokens = True,         # '[CLS]' y '[SEP]'.
								max_length = self.max_length, # Padding/truncado.
								pad_to_max_length = True,
								return_attention_mask = True,      # Attention masks.
								return_tensors = 'pt',             # Devuelve tensores de pytorch.
						  )
			
			# Agrega el texto con encoding a la lista de inputs del modelo.
			input_ids.append(encoded_dict['input_ids'])
			
			# Idem con los attention masks (sirven para identificar padding de no-padding).
			attention_masks.append(encoded_dict['attention_mask'])

		# Convierte las listas en tensores.
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		labels = torch.tensor(labels)

		return input_ids, attention_masks, labels

	def transform(self, X, y=[], sampler='random'):

		if len(y) == 0:
			y = [0]*len(X)

		input_ids, attention_masks, labels = self.tokenize(X, y)

		dataset = TensorDataset(input_ids, attention_masks, labels)

		if sampler == 'random':
			sampler = RandomSampler(dataset)
		elif sampler == 'sequential':
			sampler = SequentialSampler(dataset)

		dataloader = DataLoader(
			dataset,                       # Muestras de training.
			sampler = sampler,             # Selección de batches aleatoria.
			batch_size = self.batch_size   # Fijamos el tamaño del batch.
			)

		return dataloader

	def train(self, dataloader):

		self.model.to(self.device)

		params = list(self.model.named_parameters())

		# Número total de steps es [número de batches] x [numbero de épocas]. 
		# (Esto no es igual al nro de muestras de training).
		total_steps = len(dataloader) * self.epochs

		# Creamos el learning rate scheduler (agiliza el entrenamiento).
		scheduler = get_linear_schedule_with_warmup(
			self.optimizer,
			num_warmup_steps = 0,
			num_training_steps = total_steps
			)


		# Fijamos random seed por reproducción
		seed_val = 42
		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		if self.device == 'cuda':
			torch.cuda.empty_cache()
			torch.cuda.manual_seed_all(seed_val)

		# Acá vamos a guardar el training loss y tiempo durante el entrenamiento.
		training_stats = []

		# Hora de inicio para calcular el tiempo de entrenamiento de TODO el modelo.
		total_t0 = time.time()

		# Para cada época...
		for epoch_i in range(self.epochs):
			
			# ========================================
			#               Training
			# ========================================

			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
			print('Training...')

			# Para calcular el tiempo que le llevó a la época.
			t0 = time.time()

			# Reset the total loss for this epoch.
			total_train_loss = 0

			# Ponemos el modelo en "modo entrenamiento"
			self.model.train()

			# Iteramos sobre los batches de los datos de training...
			for step, batch in enumerate(dataloader):

				# Cada 40 batches imprimimos por pantalla el progreso.
				if step % 40 == 0 and not step == 0:
					elapsed = format_time(time.time() - t0)
					print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

				# Obtenemos los tensores del batch y los mandamos al device
				b_input_ids = batch[0].to(self.device)
				b_input_mask = batch[1].to(self.device)
				b_labels = batch[2].to(self.device)

				# Limipiamos los gradientes de la pasada backward anterior.
				self.model.zero_grad()        

				# Hacemos la pasada forward y nos quedamos con el loss y con los logits.
				loss, logits = self.model(
					b_input_ids,
					attention_mask=b_input_mask,
					labels=b_labels
					)

				# Sumamos el loss del batch para calcular el loss promedio al final.
				total_train_loss += loss.item()
				# Hacemos la pasada backward para calcular los gradientes.
				loss.backward()
				# Esto sirve para prevenir el problema de "gradientes que explotan".
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				# Actualizamos los pesos moviéndonos en dirección opuesta al gradiente.
				self.optimizer.step()
				# Actualizamos el learning rate con el scheduler.
				scheduler.step()

			# Calculamos el loss promedio sobre todos los batches.
			avg_train_loss = total_train_loss / len(dataloader)            
			
			# Medimos cuánto demoró el epoch.
			training_time = format_time(time.time() - t0)

			print("")
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
			print("  Training epcoh took: {:}".format(training_time))

		print("")
		print("Training complete!")

		print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

		# Creamos el dir de guardado si es que no existe
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		print("Saving model to %s" % self.output_dir)

		# Guardamos el modelo pre-entrenado.
		# Después se puede levantar desde la función `from_pretrained()`
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
		model_to_save.save_pretrained(self.output_dir)
		self.tokenizer.save_pretrained(self.output_dir)