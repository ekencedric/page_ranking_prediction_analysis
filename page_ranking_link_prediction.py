# -*- coding: utf-8 -*-

import json
import networkx as nx
# import matplotlib.pyplot as plt
import collections
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import girvan_newman
import itertools
from operator import itemgetter
import random

""" Creating empty graphs """

dblp2005 = nx.Graph()
dblp2006 = nx.Graph()
dblp2005w = nx.Graph()

# Graph construction from dataset
def generateGraphs():
    fileName = "dblp_coauthorship.json"
    raw_data = json.load(open(fileName))

    dblp2005 = nx.Graph()
    dblp2006 = nx.Graph()
    dblp2005w = nx.Graph()

    for content in raw_data:
        author1 = content[0]
        author2 = content[1]
        year = content[2]
        
        if year == 2005:
            dblp2005.add_edge(author1, author2)
            if dblp2005w.has_edge(author1, author2):
                dblp2005w[author1][author2]['weight'] += 1
            else:
                dblp2005w.add_edge(author1, author2, weight=1)
                
        elif year == 2006:
            dblp2006.add_edge(author1, author2)
            
    print("Graphs created!")
    return dblp2005, dblp2005w, dblp2006


""" ----Obtaining the giant connected component---- """

def gcc(G1, G2, G3):
    gcc_dblp2005 = G1.subgraph(max(nx.connected_components(G1), key=len))
    gcc_dblp2005w = G2.subgraph(max(nx.connected_components(G2), key=len))
    gcc_dblp2006 = G3.subgraph(max(nx.connected_components(G3), key=len))
    print("Giant component for graphs created!")
    return gcc_dblp2005, gcc_dblp2005w, gcc_dblp2006


dblp2005, dblp2005w, dblp2006 = generateGraphs()
gcc_dblp2005, gcc_dblp2005w, gcc_dblp2006 = gcc(dblp2005, dblp2005w, dblp2006)
# gcc_dblp2005, gcc_dblp2005w, gcc_dblp2006 = generateGraphs()

# Displaying graph info
def graphInfo(G, fileName):
    print(fileName, "is a Giant Connected", G)


"""
Getting pagerank

the names and scores of the 50 most important authors based on PageRank scores
"""
def getpageRank(G):
    pr = nx.pagerank(G)
    PRtop = sorted(pr.items(), key=lambda x: -x[1])[:50]

    for x in list(PRtop):
        print(x)


""" Getting betweenness """

def getBetweenness(G):
        bet20 = sorted(nx.edge_betweenness_centrality(
            G, k=10000).items(), key=lambda x: -x[1])[:20]
        for x in list(bet20):
            print(x)


"""GETTING STRONGLY CONNECTECT GIANT COMPONENT"""

def getCore(G):
    deg = G.degree()
    node3 = [n for n, v in dict(deg).items() if v > 2]
    return G.subgraph(node3)


""" Getting Friends Of Friends of 2005 core """

def getFOF(G):
    FOF = nx.Graph()
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            for friend_of_neighbor in G.neighbors(neighbor):
                if friend_of_neighbor not in G.neighbors(node):
                    FOF.add_edge(node, friend_of_neighbor)
    return FOF

""" Obtaining edges present in  dblp2006_core but not present in  dblp2005-core. """

def getTargetEdges(G1, G2):
    T = nx.Graph()
    for edge in G1.edges:
        if edge not in G2.edges:
            T.add_edge(*edge)

    with open('target_edges.edgelist', 'wb+') as target:
        nx.write_edgelist(T, target, delimiter=' ')


""" RD: Random Predictor """ 
def randomPredictor(G):
    
        RD = nx.Graph()
        e = list(G.edges())

        for i in range(252968):  # this value in range is the number of target edges
            edge = random.choice(e)
            # for  u, v in edge:
            #   RD.add_edge(u, v)
            RD.add_edge(*edge)
        with open('random_predictor.edgelist', 'wb+') as fw:
            nx.write_edgelist(RD, fw, delimiter=' ')

""" CN: Common Neighbors Predictor """
def commonNeighbors():
    
    CN = nx.Graph()
    for edge in FOF:
        CN.add_edge(*edge, total=sum(1 for _ in nx.common_neighbors(dblp2005_core, *edge)))
        
        return CN


""" JC: Jaccard Predictor """
def jaccardPred():

    with open("2005C.edgelist") as c:
        CORE2005 = nx.read_edgelist(c)
        JC = nx.Graph()

        with open("friends_of_friends.edgelist") as fof:
            FOF = nx.read_edgelist(fof)
            jacc = nx.jaccard_coefficient(CORE2005, ebunch=FOF.edges())

            with open("jaccard_predictor.edgelist", "wb+") as jp:
                for node1, node2, coef in jacc:
                    JC.add_edge(node1, node2, total=coef)
                    nx.write_edgelist(JC, jp, delimiter=' ')

""" PA: Preferential Attachment Predictor """
def prefAttachPred():

    with open("2005C.edgelist") as c:
        CORE2005 = nx.read_edgelist(c)
        P = nx.Graph()

        with open("friends_of_friends.edgelist") as fof:
            FOF = nx.read_edgelist(fof)
            pa = nx.preferential_attachment(CORE2005, ebunch=FOF.edges())

            with open("pref_predictor.edgelist", "wb+") as pref:
                for node1, node2, score in pa:
                    P.add_edge(node1, node2, total=score)
                    nx.write_edgelist(P, pref, delimiter=' ')

""" AA: Adamic Adar Predictor """
def adamPred():

    with open("2005C.edgelist") as c:
        CORE2005 = nx.read_edgelist(c)
        P = nx.Graph()

        with open("friends_of_friends.edgelist") as fof:
            FOF = nx.read_edgelist(fof)
            ap = nx.adamic_adar_index(CORE2005, ebunch=FOF.edges())

            with open("pref_predictor.edgelist", "wb+") as adam:
                for node1, node2, index in ap:
                    P.add_edge(node1, node2, total=index)
                    nx.write_edgelist(P, adam, delimiter=' ')

""" Getting Precision """
def getPrecision(input, k=10):
    with open(input) as file:
        G = nx.read_edgelist(file)
        total_attr = nx.get_edge_attributes(G, 'total')

        best = sorted(nx.get_edge_attributes(
            G, 'total').items(), key=lambda x: -x[1])[:k]

    with open("target_edges.egdelist") as te:
        T = nx.read_edgelist(te)

        n = 0
        for b in best:
            if T.has_edge(*b[0]):
                n += 1

        print(input, ": precision at ", k, "=", n/k)

""" Obtaining Girvan Newman """
def get_gn2(G, k):
    k = k
    comp = girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    for communities in limited:
        print(tuple(sorted(c) for c in communities))

print("\nPART A")
print("Displaying the number of nodes and edges of the Graphs")
graphInfo(gcc_dblp2005, 'gcc_dblp2005')
graphInfo(gcc_dblp2005w, 'gcc_dblp2005w')
graphInfo(gcc_dblp2006, 'gcc_dblp2006')

print("\nDisplaying pageRank and  Edge Betweeness Centrality\n")
print("gcc_dblp2005 PageRank:\n")
getpageRank(gcc_dblp2005)
print("\ngcc_dblp2005w PageRank:\n")
getpageRank(gcc_dblp2005w)
print("\ngcc_dblp2006 PageRank:\n")
getpageRank(gcc_dblp2006)


print("\n\nEDGE BETWEENESS CENTRALITY\n\n")
print("gcc_dblp2005 Edge Betweeness Centrality:\n")
getBetweenness(gcc_dblp2005)
print("\n\ngcc_dblp2005w Edge Betweeness Centrality:\n")
getBetweenness(gcc_dblp2005w)
print("\ngcc_dblp2006 Edge Betweeness Centrality:\n")
getBetweenness(gcc_dblp2006)
print("\n\nPART C: Link Prediction and Precision\n")
print("Cores:\n")
get_gn2(dblp2005, 3)

dblp2005_core = getCore(dblp2005)
dblp2006_core = getCore(dblp2006)

graphInfo(dblp2005_core, "core_2005")
graphInfo(dblp2006_core, "core_2006")

FOF = getFOF(dblp2005_core)
getTargetEdges(dblp2005_core, dblp2006_core)

randomPredictor(FOF)
CN = commonNeighbors()
# jaccardPred()
# prefAttachPred()
# adamPred()