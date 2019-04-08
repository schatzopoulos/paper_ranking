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
from math import ceil
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

    # no outlinks
    if (len(target_ids) == 1 and target_ids[0] == "0"):
        target_ids = []

    return source_id, target_ids

def collect_and_print(col):
    for x  in col.collect():
        print(x)

def pagerank_score(rank, alpha, initial_pagerank):
    return alpha * rank  + (1 - alpha) * initial_pagerank

def toCSVLine(data):
    return "\t".join(str(d) for d in data)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: pagerank <file> <alpha> <convergence_error>", file=sys.stderr)
        sys.exit(-1)


    input_file = sys.argv[1]
    
    # damping factor
    alpha = float(sys.argv[2])

    # number of iterations
    convergence_error = float(sys.argv[3])

    # initialize the spark context
    spark = SparkSession.builder.appName("SparkPageRank").getOrCreate()
    
    # supress Spark INFO messages
    log4j = spark._jvm.org.apache.log4j
    log4j.LogManager.getRootLogger().setLevel(log4j.Level.WARN)

    # read lines from file
    lines = spark.read.text(input_file).rdd.map(lambda r: r[0])
    
    partitions_count = 1       # number of executors * number of cores per executor

    # extract citation links
    links = lines.map(lambda line: parse_line(line)).partitionBy(partitions_count).cache()

    # total number of nodes
    node_count = links.count()
    print("Number of nodes: %s" % (node_count))
    print("Convergence Error: %s" % (convergence_error))
    print("Partitions: %s" % (partitions_count))

    # initialize pagerank score
    initial_pagerank = 1 / float(node_count)
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], initial_pagerank), preservesPartitioning = True)

    # initialize error in a high value
    max_error = 100
    
    iteration = 0

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    while(max_error >= convergence_error):        
        start_time = time.time()

        prev_ranks = ranks

        # find dangling nodes
        dangling_nodes = links.filter(lambda link: not link[1])

        # calculate dangling sum
        dangling_sum = dangling_nodes.join(ranks).map(lambda x: x[1][1]).sum()
        dangling_sum /= node_count

        # add dangling sum to all nodes
        dangling_contribs = links.mapValues(lambda x: dangling_sum)

        contribs = links.join(ranks).flatMap(
            lambda url_urls_rank: compute_contribs(url_urls_rank[1][0], url_urls_rank[1][1]))
        
        contribs = contribs.union(dangling_contribs)

        # print()
        # print((links.getNumPartitions(), links.partitioner))
        # print((ranks.getNumPartitions(), ranks.partitioner))

        # re-calculate pagerank score from neighbor contributions
        ranks = contribs.reduceByKey(add, numPartitions = links.getNumPartitions()).mapValues(lambda rank: pagerank_score(rank, alpha, initial_pagerank))
        # print((ranks.getNumPartitions(), ranks.partitioner))
        # calculate error between consecutive iterations
        max_error = ranks.join(prev_ranks).mapValues(lambda rank: abs(rank[0] - rank[1])).values().max()
        print("Iteration: %s - max error: %s - time: %s" % (iteration, max_error, (time.time() - start_time)))
        iteration += 1

    # collect all scores - sort desc and print
    ranks.sortBy(lambda x: - x[1]).coalesce(1).map(toCSVLine).saveAsTextFile("pagerank_results")

    spark.stop()
