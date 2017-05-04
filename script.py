import os
import pickle
from file_ids import *
from random import randint

'''
2 import pickle
   3 
   4 favorite_color = { "lion": "yellow", "kitty": "red" }
   5 
   6 pickle.dump( favorite_color, open( "save.p", "wb" ) )
   '''

#Preprocessing token_numbers in Vlad's file to have cumulative counts of tokens
# for f in os.walk("./parsed"):	
# 	all_files = f[2]
# 	for x in all_files:
# 		if 'parsed' in x:	
# 			if x[0:x.find('.')] in ret_test_fnames():
#  				continue		
# 			token_num = 0
# 			token_id = 1
# 			#sent_id = 1
# 			out = x + "_output.txt"
# 			output = open(os.path.join("parsed_output/",out),"w+")
# 			with open("parsed/"+x,"r") as file:
# 				for line in file:					
# 					current = line.split("	")					
# 					if len(current)  != 1:						
# 						token_num = token_num + 1
# 						output.write(str(token_num)+"	"+current[1]+"		"),
# 						# output.write(str(sent_id)+"		"+str(token_id))
# 						#mapping[(sent_id,token_id)] = token_num
# 						#token_id = token_id + 1
# 						for count in range(2,len(current)-1):
# 						 	output.write(current[count]+"	"),
# 						output.write(current[len(current)-1]),
# 						# output.write("\n")
# 					else:
# 						output.write(line)


#To find token number given (sent_id,token_id)
# mapping = {}
# for f in os.walk("./parsed"):	
# 	all_files = f[2]
# 	for x in all_files:
# 		if 'parsed' in x:	
# 			if x[0:x.find('.')] in ret_test_fnames():
#   				continue		
# 			token_num = 0
# 			token_id = 0
# 			sent_id = 0
# 			out = x + "_output.txt"
# 			output = open(os.path.join("parsed_mapping/",out),"w+")
# 			with open("parsed/"+x,"r") as file:
# 				for line in file:					
# 					current = line.split("	")					
# 					if len(current)  != 1:						
# 						token_num = token_num + 1
# 						# output.write(str(token_num)+"	"+current[1]+"		"),
# 						#output.write(str(sent_id)+"		"+str(token_id)+"		"+str(token_num))
# 						mapping[(sent_id,token_id)] = token_num
# 						token_id = token_id + 1
# 						# for count in range(1,len(current)-1):
# 						# 	output.write(current[count]+"	"),
# 						#output.write(current[len(current)-1]),
# 						#output.write("\n")
# 					else:
# 						#output.write(line)
# 						token_id = 0
# 						sent_id = sent_id + 1
# 				output.write(str(mapping))
			

#Making list of word_types and indices for each of them
all_tokens_to_index = {}
all_index_to_tokens = {}
all_tokens_to_index_unk = {}
all_index_to_tokens_unk = {}
index = 0
freq_unk = 0
freq = {}
for f in os.walk("./parsed"):	
	all_files = f[2]
	for x in all_files:
		if 'parsed' in x:			
			if x[0:x.find('.')] in ret_test_fnames():
				continue
			token_num = 0			
			with open("parsed/"+x,"r") as file:
				for line in file:					
					current = line.split("	")
					if len(current)  != 1:
						word = current[1]
						if word not in freq:
							freq[word] = 1
						else:
							freq[word] = freq[word] + 1
						
						if word not in all_tokens_to_index:
							all_tokens_to_index[word] = [index,freq[word]]
							all_index_to_tokens[index] = word
							index = index+1
						else:
							all_tokens_to_index[word][1] = freq[word]

#print index
all_tokens_to_index['UNK'] = [index,1]   	#13228
all_tokens_to_index['NONE'] = [index+1,1]		#13229
print len(all_tokens_to_index)		#13230
#count = 0
#tot = 0
curr = 2

for x in all_tokens_to_index:
	# print " "
	# print all_tokens_to_index[x]
	if all_tokens_to_index[x][1]==1:
		#tot= tot + 1
		r = randint(0,1000)
		if r<5:
			freq_unk = freq_unk + 1
			all_tokens_to_index_unk[x] = [0,freq_unk]
			#all_index_to_tokens[curr] = 'UNK'
			#curr = curr + 1
		else:
			all_tokens_to_index_unk[x]= [curr,all_tokens_to_index[x][1]]
			all_index_to_tokens_unk[curr] = x
			curr = curr + 1
	else:
		all_tokens_to_index_unk[x] = [curr,all_tokens_to_index[x][1]]
		all_index_to_tokens_unk[curr] = x
		curr = curr + 1
all_tokens_to_index_unk['UNK'] = [0,freq_unk]
all_index_to_tokens_unk[0] = 'UNK'
all_tokens_to_index_unk['None'] = [1,1]
all_index_to_tokens_unk[1] = 'None'
print current		#13198
#print all_index_to_tokens_unk[1300]


pickle.dump( all_tokens_to_index_unk, open( "vocab.txt", "wb" ) )

# print index
# print "TOKENS TO INDEX,FREQ"
# print all_tokens_to_index
# print "			"
# print "			"
# print "INDEX TO TOKENS"
# print all_index_to_tokens



#Printing index number for token in every file
for f in os.walk("./parsed"):	
	all_files = f[2]
	for x in all_files:	
		if 'parsed' in x:
			# if x[0:x.find('.')] in ret_test_fnames():
 		# 		continue
			out = x+"_indices.txt"	
			output = open(os.path.join("parsed_indices/",out),"w+")			
			with open("parsed/"+x,"r") as file:
				for line in file:					
					current = line.split("	")					
					if len(current)  != 1:	
						ind = 0					
						try:
							ind = all_tokens_to_index_unk[current[1]][0]
						except:
							ind = all_tokens_to_index_unk['UNK'][0]
						print ind
						output.write(str(ind))
						output.write("\n")
					else:
						output.write(line)
						
