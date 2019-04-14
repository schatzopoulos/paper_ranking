#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# nohup spark2-submit --driver-memory 5g --executor-memory 10G --executor-cores 5 --num-executors 7 --conf spark.yarn.executor.memoryOverhead=1024 sim_pagerank.py hdfs://master2:8020/data/dblp-papers/dblp_oldest_50_percent.txt hdfs://master2:8020/data/dblp-papers/sim_w3.csv 0.5 0.25 0.25 0.000000000001 0.6 &
"""
This is an example implementation of PageRank. For more conventional use,
Please refer to PageRank implementation provided by graphx

Example Usage:
bin/spark-submit examples/src/main/python/pagerank.py data/mllib/pagerank_data.txt 10
"""
from __future__ import print_function

import re
import sys
import time
import itertools
import subprocess
from operator import add

from pyspark.sql import SparkSession


def compute_contribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    
    for url in urls:
        yield (url, rank / num_urls)

def parse_neighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]

def parse_line(line):
    """Parses a urls pair string into urls pair."""

    # split in tabs
    parts = re.split(r'\s+', line)
    source_id = parts[0]
    target_ids = parts[1].split("|")[0].split(",")
    year = int(parts[3])

    # no outlinks
    if (len(target_ids) == 1 and target_ids[0] == "0"):
        target_ids = []

    return source_id, (target_ids, year)

def parse_sim_line(line): 
    parts = re.split(r'\s+', line)
    return parts[0], (parts[1], float(parts[2]))

def parse_cc_line(line): 
    """Parses a urls pair string into urls pair."""

    # split in tabs
    parts = re.split(r'\s+', line)
    source_id = parts[0]
    target_ids = parts[1].split("|")[0].split(",")
    year = int(parts[3])

    # no outlinks
    if (len(target_ids) == 1 and target_ids[0] == "0"):
        target_ids = []

    citations = []
    for t in target_ids: 
        citations.append((t, source_id, year))

    return citations

def collect_and_print(col):
    for x  in col.collect():
        print(x)

def pagerank_score(rank, alpha, initial_pagerank, beta, cc, gamma):
    return alpha * rank  + beta * initial_pagerank + gamma * cc

def toCSVLine(data):
    return "\t".join(str(d) for d in data)

def count_citations(lines, links, min_year):
   
    # collect citations after min_year
    cc = lines.flatMap(lambda line: parse_cc_line(line)).filter(lambda x: x[2] >= min_year)

    # sum citations per paper after min_year
    cc = cc.map(lambda x: (x[0], 1)).reduceByKey(add)

    # fill cc with 0 for remaining papers 
    cc = links.leftOuterJoin(cc, numPartitions = links.getNumPartitions()).mapValues(lambda x: x[1] if x[1] is not None else 0)

    return cc

def calculate_similarity_cc(sim_file, min_sim_score, cold_start_year, links, cc):

    # find papers in cold start period
    cold_start_papers = links.filter(lambda x: x[1][1] >= cold_start_year)

    # read similarities from file 
    sim_lines = spark.read.text(sim_file).rdd.map(lambda r: r[0])

    # parse file with similarities
    similarities = sim_lines.map(lambda line: parse_sim_line(line))
    
    # keep similarities only for papers in cold start && with similarity score greater than min_sim_score
    similarities = similarities.filter(lambda r: r[1][1] >= min_sim_score).join(cold_start_papers).mapValues(lambda x: x[0])

    # here we have tuples e.g. ('e', ('b', 0.9)), ('e', ('d', 0.2)), ('f', ('c', 1.0)) etc
    # for first tuple meaning that: paper(e) is similar to paper(b) with score 0.9
    # we move similar paper ids to keys in order to join, we also discard similarity score
    similarities = similarities.map(lambda x: (x[1][0], x[0]))
    
    # join similar papers with citation counts and keep (cold start) papers and 
    # the citation count of each one of their similar papers
    similarities = similarities.join(cc).map(lambda x: (x[1][0], x[1][1]))

    # calculate average (see https://stackoverflow.com/a/29930162)
    aTuple = (0,0)
    similarities = similarities.aggregateByKey(aTuple, lambda a,b: (a[0] + b,    a[1] + 1),
                                       lambda a,b: (a[0] + b[0], a[1] + b[1]))

    similarities = similarities.mapValues(lambda v: v[0]/v[1])

    return similarities

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: pagerank <file> <sim_file>", file=sys.stderr)
        sys.exit(-1)
    combinations = list(itertools.product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], repeat=3))

    for comb in combinations:

        comb_sum = comb[0] + comb[1] + comb[2]
        if (comb_sum <= 0.95 or comb_sum >= 1.05):
            continue;
        
        for min_year_var in range(0, 11):
            for cold_year_var in range(0, 6):
                alpha = comb[0]
                beta = comb[1]
                gamma = comb[2]
                convergence_error = 0.00000000001
                min_sim_score = 0.6

                
                
                input_file = sys.argv[1]
                
                sim_file = sys.argv[2]

                job_name = "SparkPageRank_a_" + str(alpha) + "_b_" + str(beta) + "_g_" + str(gamma) + "_my_" + str(min_year_var) +  "_cy_" + str(cold_year_var)

                # initialize the spark context
                spark = SparkSession.builder.appName(job_name).getOrCreate()
                
                # supress Spark INFO messages
                log4j = spark._jvm.org.apache.log4j
                log4j.LogManager.getRootLogger().setLevel(log4j.Level.WARN)

                # read lines from file
                lines = spark.read.text(input_file).rdd.map(lambda r: r[0])
                
                partitions_count = 35       # number of executors * number of cores per executor

                # extract citation links
                links = lines.map(lambda line: parse_line(line)).partitionBy(partitions_count).cache()

                max_year = links.map(lambda x: x[1][1]).max()

                # count citations after that year
                min_year = max_year - min_year_var

                # handle papers after that year as papers in cold start
                cold_start_year = max_year - cold_year_var

                # calculate citation counts for last X years
                cc = count_citations(lines, links, min_year)

                # calculate similarities
                cc_sim = calculate_similarity_cc(sim_file, min_sim_score, cold_start_year, links, cc)
                
                # merge citation count computed based on similarity
                cc = cc.leftOuterJoin(cc_sim, numPartitions = links.getNumPartitions()).mapValues(lambda x: x[1] if x[1] is not None else x[0]).cache()
                
                # normalize cc values
                max_value = cc.values().sum()

                # avoid division by zero when all cc scrores are zero
                if (max_value == 0):
                    max_value = 1

                cc = cc.mapValues(lambda x: x / float(max_value))

                # total number of nodes
                node_count = links.count()
                # print("Number of nodes: %s" % (node_count))
                # print("Convergence Error: %s" % (convergence_error))
                # print("Alpha: %s" % (alpha))
                # print("Beta: %s" % (beta))
                # print("Gamma: %s" % (gamma))
                # print("Min score: %s" % (min_sim_score))
                # print("Year: %s" % (years_in_cold_start))

                # print("Partitions: %s" % (partitions_count))

                # initialize pagerank score
                initial_pagerank = 1 / float(node_count)
                ranks = links.map(lambda url_neighbors: (url_neighbors[0], initial_pagerank), preservesPartitioning = True)

                # initialize error in a high value
                max_error = 100
                
                iteration = 0
                print((alpha, beta, gamma, years_in_cold_start))
                # Calculates and updates URL ranks continuously using PageRank algorithm.
                while(max_error >= convergence_error):        
                    start_time = time.time()
                    prev_ranks = ranks

                    # find dangling nodes
                    dangling_nodes = links.filter(lambda link: not link[1][0])

                    # calculate dangling sum
                    dangling_sum = dangling_nodes.join(ranks).map(lambda x: x[1][1]).sum()
                    dangling_sum /= node_count

                    # add dangling sum to all nodes
                    dangling_contribs = links.mapValues(lambda x: dangling_sum)
                    
                    contribs = links.join(ranks, numPartitions = links.getNumPartitions()).flatMap(
                        lambda url_urls_rank: compute_contribs(url_urls_rank[1][0][0], url_urls_rank[1][1]))

                    contribs = contribs.union(dangling_contribs).coalesce(links.getNumPartitions())

                    # re-calculate pagerank score from neighbor contributions
                    ranks = contribs.reduceByKey(add, numPartitions = links.getNumPartitions())

                    # na tsekarw oti to sum tou dianumatos a = 1, b = 1 kai g = 1, to dianisma sum ston asso
                    ranks = ranks.join(cc, numPartitions = links.getNumPartitions()).mapValues(lambda x: pagerank_score(x[0], alpha, initial_pagerank, beta, x[1], gamma))   

                    # calculate error between consecutive iterations
                    max_error = ranks.join(prev_ranks).mapValues(lambda rank: abs(rank[0] - rank[1])).values().max()

                    print("\tIteration: %s - max error: %s - time: %s" % (iteration, max_error, (time.time() - start_time)))
                    iteration += 1

                ranks.sortBy(lambda x: - x[1]).coalesce(1).map(toCSVLine).saveAsTextFile(job_name)

                spark.stop()
