Code :
data : entire data, has all the documents from BeSt
parsed_output : cumulative counts
parsed_mapping : given sent_id,token_id, returns token number
parsed_indices : index for the token in that doc, which comes from the big vocab.txt dictionary
script.py : Index,tokens, all mappings
parsed : has the outputs from running Stanford Parser 
loaders.py : generates parsed(for syntax of this talk to Vlad for help)
file_ids.py : has all the file ids, split into train and test (has functions to extract accordingly)
pairs_version1.py : creates all possible combinations of pairs
util : has vocabmapping.py for dictionary reading/dumping done with pickle
models : contains sentiment.py which is the main LSTM model
train.py : gets batches of train and test and calls model.step on them(session run)
config.ini : has hyperparameters for the model
wordtovec.py : gives the word2vec embeddings
prec.py : used to calculate precision, recall and F1 scores



TO RUN:
Create directories named parsed_indices, parsed_mapping and parsed_output
Run python script.py
Execute script.py
Download tensorflow 1.0.0
Create a directory named pairs and Run python pairs_version1.py
Then run python train.py



