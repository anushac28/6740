import os

#Preprocessing token_numbers
'''
for f in os.walk("./parsed"):	
	all_files = f[2]
	for x in all_files:
		if 'parsed' in x:			
			token_num = 0
			out = x + "_output.txt"
			output = open(out,"w+")
			with open("parsed/"+x,"r") as file:
				for line in file:					
					current = line.split("	")					
					if len(current)  != 1:						
						token_num = token_num + 1
						output.write(str(token_num)+"	"),
						for count in range(1,len(current)-1):
							output.write(current[count]+"	"),
						output.write(current[len(current)-1]),
					else:
						output.write(line)
'''				

#Making list of word_types
all_tokens = {}
index = 0
freq = {}
for f in os.walk("./parsed"):
	#if 'parsed' in f:
	# print f
	# print " "
	all_files = f[2]
	for x in all_files:
		if 'parsed' in x:
			#print x
			token_num = 0
			#out = x + "_output.txt"
			#output = open(out,"w+")
			with open("parsed/"+x,"r") as file:
				for line in file:					
					current = line.split("	")
					if len(current)  != 1:
						word = current[1]
						if word not in freq:
							freq[word] = 1
						else:
							freq[word] = freq[word] + 1
						
						if word not in all_tokens:
							all_tokens[word] = [index,freq[word]]
							index = index+1
						else:
							all_tokens[word][1] = freq[word]

print all_tokens
					
