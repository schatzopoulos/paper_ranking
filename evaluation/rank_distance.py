#!/opt/anaconda/bin/python

# Program to calculate spearman's rho,
# using the Bio.Cluster module of python

# -------------------------------------------------------------------- #
# Imports
import sys
import os
import Bio.Cluster
import scipy
import scipy.stats
import subprocess

# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# Function definitions

# Function that returns the rank scores of papers in a paper id sorted file
def read_file_ranks(file_name, pub_id_dict = None):
	# List containing ranks. To be returned
	rank_list = list()
	# Read file contents - assumes a file SORTED by PAPER ID
	contents = open(file_name, "r").readlines()
	contents = [line.strip() for line in contents]

	#counter = 0
	
	# Read ranks into list - optionally exclude some
	if(not pub_id_dict):
		for content in contents:
			# print "File1, adding: ", content.split()[1]
			rank_list.append(content.split()[1])
			#if(counter < 10):
			#	print content.split()[0], content.split()[1]
			#counter+=1
	# Exclude those papers not found previously
	else:
		for content in contents:
			line_parts = content.split()
			pid = line_parts[0]
			score = line_parts[1]
			if(pid in pub_id_dict):
				# print "File2, adding: ", score
				rank_list.append(score)
				#if(counter < 10):
				#	print content.split()[0], content.split()[1]
				#counter+=1
			else:
				pass
		
	return rank_list
	
# Function to read valid Paper IDs (those in older file)
def get_valid_paper_ids(file_name):
	valid_papers = dict()
	print(file_name)
	contents = open(file_name, "r").readlines()
	# print(contents)
	contents = [line.strip().split()[0] for line in contents]
	# print(contents)
	for content in contents:
		valid_papers[content] = 1
	
	return valid_papers

# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# Reading of arguments

article_file_old = sys.argv[1]
article_file_newer = sys.argv[2]
metric = sys.argv[3]
get_common = False
if(metric.startswith("k")):
	metric = "k"
else:
	metric = "s"
try:
	if(sys.argv[4] != None):
		get_common = True
except:
	pass

# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# Main program
print(get_common)
# First sort files by paper id
if(not get_common):
	subprocess.call("sort -k1,1 \"" + article_file_old + "\" > older_file_sorted.txt", shell=True)
	subprocess.call("sort -k1,1 \"" + article_file_newer + "\" > newer_file_sorted.txt", shell=True)
else:
	subprocess.call("cut -f 1 \"" + article_file_old + "\" | sort > file1_sorted.txt", shell=True)
	subprocess.call("cut -f 1 \"" + article_file_newer + "\" | sort > file2_sorted.txt", shell=True)
	# Get common lines
	subprocess.call("comm -12 file1_sorted.txt file2_sorted.txt > common_lines.txt", shell=True)
	subprocess.call("sort -k1,1 \"" + article_file_old + "\" | join -j 1 common_lines.txt - | sed 's/ /\t/' > older_file_sorted.txt", shell=True)
	subprocess.call("sort -k1,1 \"" + article_file_newer + "\" | join -j 1 common_lines.txt - | sed 's/ /\t/' > newer_file_sorted.txt", shell=True)

# print(os.popen("head older_file_sorted.txt").read())
# Get valid papers (existing in both files)
valid_papers = get_valid_paper_ids("older_file_sorted.txt")
# print(valid_papers)
# print out first 10 of each, just to check
# print os.popen("head -10 older_file_sorted.txt").read()
# print os.popen("head -10 newer_file_sorted.txt").read()

# print "Valid Papers are:", valid_papers

# Then read ranks into list
# a. Read all paper_id + ranks from older file
old_file_list = read_file_ranks("older_file_sorted.txt")
# print "Old list: ", old_file_list

# b. Read paper_id's from second file that are in older file
new_file_list = read_file_ranks("newer_file_sorted.txt", valid_papers)
# print "New list: ", new_file_list
# print(old_file_list)
# print old_file_list[:10]
# print new_file_list[:10]

if(metric == "s"):
	# Fees rank lists into Bio.Cluster method
	spearman_dist = Bio.Cluster.distancematrix((old_file_list,new_file_list), dist=metric)[1][0]
	# Output rho
	print ("Rho: " + str(1-spearman_dist))
	print ("Dist:" + str(spearman_dist))
	
elif(metric == "k"):
	kendall_dist = Bio.Cluster.distancematrix((old_file_list,new_file_list), dist=metric)[1][0]
	scipy_kendall = scipy.stats.stats.kendalltau(old_file_list, new_file_list)[0]
	# Output Kentall's tau
	print ("Dist:" + str(kendall_dist))
	print ("Tau: " + str(1-kendall_dist))
	print ("Scipy Tau: " + str(scipy_kendall))

# Remove sorted files
if(not get_common):
	os.popen("rm older_file_sorted.txt")
	os.popen("rm newer_file_sorted.txt")
else:
	os.popen("rm file1_sorted.txt file2_sorted.txt older_file_sorted.txt newer_file_sorted.txt common_lines.txt")

# -------------------------------------------------------------------- #
