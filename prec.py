#tf.contrib.metrics.streaming_precision
# number = 1
# positives = []
# with open("3.txt",'r') as f:
# 	for line in f:
# 		if "[0, 1, 0]" in line or "[0, 0, 1]" in line:
# 			#print number
# 			positives.append(number)
# 		number = number + 1
# #print positives

# number = 1
# predicted_pos = []
# with open("2.txt",'r') as f:
# 	for line in f:
# 		if number not in positives and ("1" in line or "2" in line):		#in gives true, not in gives false
# 			#print number
# 			predicted_pos.append(number)
# 		number = number + 1
# print positives
# print predicted_pos
# print len(positives)
# print len(predicted_pos)	#95 = tp

#PREDICTED IS 2.txt
#ACTUAL is 3.txt

# number = 1
# negatives = []
# with open("3.txt",'r') as f:
# 	for line in f:
# 		if "[1, 0, 0]" in line:
# 			#print number
# 			negatives.append(number)
# 		number = number + 1
# #print negatives		#103731
# print len(negatives)

# number = 1
# pred_negatives = []
# with open("2.txt",'r') as f:
# 	for line in f:
# 		if number not in negatives and "0" in line:
# 			pred_negatives.append(number)
# 		number = number + 1

# print len(pred_negatives)

