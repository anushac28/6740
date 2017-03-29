Code :
data : entire data
parsed : Vlad's files
parsed_output : cumulative counts
parsed_mapping : given sent_id,token_id, returns token number
parsed_indices : index for the token in that doc, which comes from the big vocab.txt dictionary
vocabulary : python script.py >vocabulary.txt
script.py : Index,tokens, all mappings

config.ini : made batch size as 1
train.py



Note :
TODO 1 -
'010aaf594ae6ef20eb28e3ee26038375'
If we look at source file we see 2 authors : randman and patrick1000
They are both present in ere
But none of them have come in the conll8_parsed/json and hence not in pairs, so we have to add a separate script to add them

TODO 2 -
Make vocab.txt only from the first 173 files and make some of the words with frequency 1 as UNK
For 73 test files, we find the index from the trained vocab.txt else use index of UNK
