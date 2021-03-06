'''
I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and semi-based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html


'''
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import sys
import math
import os
import ConfigParser
import random
import time
from six.moves import xrange
#import util.dataprocessor
import util.hyperparams as hyperparams
import models.sentiment
import util.vocabmapping
import file_ids
from file_ids import *
from file_ids2 import * 
import numpy as np 

blog_t = ret_train_fnames()
blog_test = ret_test_fnames()
#blog_t = ['0a421343005f3241376fa01e1cb3c6fb','0ba982819aaf9f5b94a7cebd48ac6018','0c49bb860962aa0d5b8e3fc277592da0','0cfdfe102b7a4cb34e1a181c1d36d23d']
#blog_test = ['1b268b27094ba9c5feb11192dad940ab','1d16a571f14fb1032bc19e9314a46deb']

def index_list(blog,flag):
	list_of_lists = []
	seqs = []
	list_of_ylabels = []
	list_of_pairs = []
	for x in blog:#blog_ids:
	#x = blog_ids[0]	
		lists = []
		pairlist = []
		ylabel = []
		try:
			with open(os.path.join("/home/anusha/Desktop/CURRENT/parsed_indices/",x+".cmp.txt.conll8.parsed_indices.txt"),'r') as f1:
				seq_length = 0
				if flag==0:
					lists.append(1)
					lists.append(0)
				for line in f1:
					if line[0]!='\n':
						lists.append(int(line.strip()))
						seq_length += 1
			#lists = np.expand_dims(lists, axis=0)
			list_of_lists.append(lists)
			seqs.append([seq_length])
		except:
			continue

		
		# with open(os.path.join("/home/anusha/Desktop/6740 Project/CURRENT/pairs_y/",x+"_pairlist.txt"),'r') as f2:
		# 	#print f2
		# 	for line in f2.readlines():
		# 		#print line
		# 		ylabel.append(line[:-1])
		# list_of_ylabels.append(ylabel)
		# f2.close()

		
		try:
			with open(os.path.join("/home/anusha/Desktop/CURRENT/pairs/",x+"_pairlist.txt"),'r') as f3:
				for line in f3.readlines():
					#print line
					if line[0]!='\n':
						pairs = line.split()					
						#print pairs
						if pairs[0]=='None':
							a = 0
						else:
							a = int(pairs[0])
						if pairs[1]=='None':
							b = 0
						else:
							b = int(pairs[1])
						if pairs[2]=='None':
							c = 0
						else:
							c = int(pairs[2])
						if pairs[3]=='None':
							d = 0
						else:
							d = int(pairs[3])
						pairlist.append([ [a,b] , [c,d] ])
						
						temp = pairs[4:]
						result = [1,0,0]
						if len(temp)>1 :
							e = temp[1]
							if 'neg' in e:
								result = [0,0,1]	#2
							if 'pos' in e:
								result = [0,1,0]	#1
						else:
							result = [1,0,0]
						ylabel.append(result)
			list_of_pairs.append(pairlist)
			list_of_ylabels.append(ylabel)
		except:
			continue
		#f3.close()

	return list_of_lists,seqs,list_of_pairs,list_of_ylabels


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")
flags.DEFINE_string("data_dir", "data/", "Path to main data directory.")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")

def main():
	hyper_params = check_get_hyper_param_dic()
	#util.dataprocessor.run(hyper_params["max_seq_length"],hyper_params["max_vocab_size"])

	#create model
	# print "Creating model with..."
	# print "Number of hidden layers: {0}".format(hyper_params["num_layers"])
	# print "Number of units per layer: {0}".format(hyper_params["hidden_size"])
	# print "Dropout: {0}".format(hyper_params["dropout"])

	train_x,train_seq,train_y_pairs,train_y_labels = index_list(blog_t,0)
	test_x,test_seq,test_y_pairs,test_y_labels = index_list(blog_test,1)
	# print train_x
	# print "*****"
	# print train_y_pairs
	# print "*****"
	# print train_y_labels

	train_data = (train_x , train_seq, train_y_pairs , train_y_labels)
	#print len(train_data[0][0])
	#print train_data[0][:][:][1]
	test_data = (test_x, test_seq, test_y_pairs, test_y_labels)
	#test _ data ???
	
	




	vocabmapping = util.vocabmapping.VocabMapping()
	vocab_size = vocabmapping.getSize()
	#pair_size = ???
	#print "Vocab size is: {0}".format(vocab_size)
	num_batches = 1
	# path = os.path.join(FLAGS.data_dir, "processed/")
	# infile = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	# #randomize data order
	# print infile
	# data = np.load(os.path.join(path, infile[0]))
	# for i in range(1, len(infile)):
	# 	data = np.vstack((data, np.load(os.path.join(path, infile[i]))))
	# np.random.shuffle(data)
	# #data = data[:3000]
	# num_batches = len(data) / hyper_params["batch_size"]
	# # 70/30 splir for train/test
	# train_start_end_index = [0, int(hyper_params["train_frac"] * len(data))]
	# test_start_end_index = [int(hyper_params["train_frac"] * len(data)) + 1, len(data) - 1]
	# print "Number of training examples per batch: {0}, \
	# \nNumber of batches per epoch: {1}".format(hyper_params["batch_size"],num_batches)
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("/tmp/tb_logs", sess.graph)
		model = create_model(sess, hyper_params, vocab_size)
	# #train model and save to checkpoint
	# 	print "Beggining training..."
	# 	print "Maximum number of epochs to train for: {0}".format(hyper_params["max_epoch"])
	# 	print "Batch size: {0}".format(hyper_params["batch_size"])
	# 	print "Starting learning rate: {0}".format(hyper_params["learning_rate"])
	# 	print "Learning rate decay factor: {0}".format(hyper_params["lr_decay_factor"])

		step_time, loss = 0.0, 0.0
		previous_losses = []
		tot_steps = num_batches * hyper_params["max_epoch"]
		model.initData(train_data, test_data)



		#print tot_steps
	# 	#starting at step 1 to prevent test set from running after first batch
		for step in xrange(1, tot_steps):
			#print step
			# Get a batch and make a step.
			start_time = time.time()

			inputs, pairs, targets, seq_lengths = model.getBatch()
			str_summary, step_loss, _ = model.step(sess, inputs, pairs, targets, seq_lengths, False)

			step_time += (time.time() - start_time) / hyper_params["steps_per_checkpoint"]
			loss += step_loss / hyper_params["steps_per_checkpoint"]
			#print "loss is ",
			#print (step_loss / hyper_params["steps_per_checkpoint"])

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if step % hyper_params["steps_per_checkpoint"] == 0:
				writer.add_summary(str_summary, step)
				# Print statistics for the previous epoch.
				# print ("global step %d learning rate %.7f step-time %.2f loss %.4f"
				# % (model.global_step.eval(), model.learning_rate.eval(),
				# step_time, loss))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				step_time, loss, test_accuracy = 0.0, 0.0, 0.0
				# Run evals on test set and print their accuracy.
				#print "Running test set"
				print "test data length"
				print len(test_data)
				for test_step in xrange(5):
					inputs, pairs, targets, seq_lengths = model.getBatch(False)
					#print targets
					i = 0
					for t in targets:
						#print pairs[i]
						print t
						i = i+1
					str_summary, test_loss, _, accuracy = model.step(sess, inputs, pairs, targets, seq_lengths, True)
					# print "details"
					# print test_step
					# print _
					# print tf.argmax(_,1)
					# for x in _:
					# 	if x[0]>x[1] and x[0]>x[2]:
					# 		print "0"
					# 	if x[1]>x[0] and x[1]>x[2]:
					# 		print "1"
					# 	if x[2]>x[1] and x[2]>x[0]:
					# 		print "2"
					loss += test_loss
					test_accuracy += accuracy
					#print "accuracy is "
					#print accuracy
				exit()
				normalized_test_loss, normalized_test_accuracy = loss / len(blog_test), test_accuracy / len(test_data)
				checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "sentiment{0}.ckpt".format(normalized_test_accuracy))
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				writer.add_summary(str_summary, step)
				#print "Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(normalized_test_loss, normalized_test_accuracy)
				print "-------Step {0}/{1}------".format(step,tot_steps)
				loss = 0.0
				sys.stdout.flush()

def create_model(session, hyper_params, vocab_size):
	model = models.sentiment.SentimentModel(vocab_size = vocab_size,
											hidden_size = hyper_params["hidden_size"],
											dropout = hyper_params["dropout"],
											num_layers = hyper_params["num_layers"],
											max_gradient_norm = hyper_params["grad_clip"],
											learning_rate = hyper_params["learning_rate"],
											lr_decay = hyper_params["lr_decay_factor"],
											batch_size = hyper_params["batch_size"],session=session)
# 	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
# 	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
# 		print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
# 		model.saver.restore(session, ckpt.model_checkpoint_path)
# 	else:
# 		print "Created model with fresh parameters."
# 		session.run(tf.initialize_all_variables())
	return model

def read_config_file():
	config = ConfigParser.ConfigParser()
	config.read(FLAGS.config_file)
	dic = {}
	sentiment_section = "sentiment_network_params"
	general_section = "general"
	dic["num_layers"] = config.getint(sentiment_section, "num_layers")
	dic["hidden_size"] = config.getint(sentiment_section, "hidden_size")
	dic["dropout"] = config.getfloat(sentiment_section, "dropout")
	dic["batch_size"] = config.getint(sentiment_section, "batch_size")
	dic["train_frac"] = config.getfloat(sentiment_section, "train_frac")
	dic["learning_rate"] = config.getfloat(sentiment_section, "learning_rate")
	dic["lr_decay_factor"] = config.getfloat(sentiment_section, "lr_decay_factor")
	dic["grad_clip"] = config.getint(sentiment_section, "grad_clip")
	dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
		"use_config_file_if_checkpoint_exists")
	dic["max_epoch"] = config.getint(sentiment_section, "max_epoch")
	dic ["max_vocab_size"] = config.getint(sentiment_section, "max_vocab_size")
	dic["max_seq_length"] = config.getint(general_section,
		"max_seq_length")
	dic["steps_per_checkpoint"] = config.getint(general_section,
		"steps_per_checkpoint")
	return dic

def check_get_hyper_param_dic():
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
	hyper_params = read_config_file()
	if serializer.checkExists():
		if serializer.checkChanged(hyper_params):
			if not hyper_params["use_config_file_if_checkpoint_exists"]:
				hyper_params = serializer.getParams()
				print "Restoring hyper params from previous checkpoint"
			else:
				new_checkpoint_dir = "{0}_hidden_size_{1}_numlayers_{2}_dropout_{3}".format(
				int(time.time()),
				hyper_params["hidden_size"],
				hyper_params["num_layers"],
				hyper_params["dropout"])
				new_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir,
					new_checkpoint_dir)
				os.makedirs(new_checkpoint_dir)
				FLAGS.checkpoint_dir = new_checkpoint_dir
 				serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
				serializer.saveParams(hyper_params)
		else:
			print "No hyper parameter changed detected"
	else:
		serializer.saveParams(hyper_params)
		print "No hyper params detected at checkpoint"
	return hyper_params

if __name__ == '__main__':
	main()

# ****
# 97
# [[ 0.15581135  0.19333872  0.65084994]
#  [ 0.31233302  0.26671427  0.42095268]
#  [ 0.24155505  0.24156073  0.51688427]
#  ..., 
#  [ 0.21081246  0.53051299  0.25867456]
#  [ 0.3341853   0.3687433   0.29707137]
#  [ 0.45486385  0.29041544  0.25472078]]
# accuracy is 
# 0.368432

