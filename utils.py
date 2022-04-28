import torch
import json
import os
import sys
import numpy as np
import pickle
from collections import Counter
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm
from ortools.graph import pywrapgraph


def readdata(datapath, mode="train"):
    print("Tokenizing sentence and word...")
    with open(datapath) as fin:
        content = json.load(fin)
    print("{} {} data examples read.".format(len(content), mode))
    all_utterances = []
    labels = []
    for item in tqdm(content):
        utterance_list = []
        label_list = []
        for one_uttr in item:
            uttr_content = one_uttr['utterance']
            label = one_uttr['label']
            label_list.append(label)
            utterance_list.append(uttr_content)
        all_utterances.append(utterance_list)
        labels.append(label_list)
    return all_utterances, labels


def readdata2(datapath, mode="train"):
    print("Tokenizing sentence and word...")
    with open(datapath) as fin:
        content = json.load(fin)
    print("{} {} data examples read.".format(len(content), mode))
    all_utterances = []
    labels = []
    speakers = []
    for item in tqdm(content):
        utterance_list = []
        label_list = []
        speaker_list = []
        for one_uttr in item:
            uttr_content = one_uttr['utterance']
            label = one_uttr['label']
            speaker = one_uttr['speaker']
            label_list.append(label)
            speaker_list.append(speaker)
            utterance_list.append(uttr_content)
        all_utterances.append(utterance_list)
        labels.append(label_list)
        speakers.append(speaker_list)
    return all_utterances, labels, speakers


def calculate_purity_scores(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def calculate_shen_f_score(y_true, y_pred):
    def get_f_score(i, j, n_i_j, n_i, n_j):
        recall = n_i_j / n_i
        precision = n_i_j / n_j
        if recall == 0 and precision == 0:
            f_score = 0.
        else:
            f_score = 2 * recall * precision / (recall + precision)
        return f_score

    y_true_cnt = dict(Counter(y_true))
    y_pred_cnt = dict(Counter(y_pred))
    y_pred_dict = dict()
    for i, val in enumerate(y_pred):
        if y_pred_dict.get(val, None) == None:
            y_pred_dict[val] = dict()
        if y_pred_dict[val].get(y_true[i], None) == None:
            y_pred_dict[val][y_true[i]] = 0
        y_pred_dict[val][y_true[i]] += 1
    shen_f_score = 0.
    for i, val_i in y_true_cnt.items():
        f_list = []
        for j, val_j in y_pred_cnt.items():
            f_list.append(get_f_score(i, j, y_pred_dict[j].get(i, 0), val_i, val_j))
        shen_f_score += max(f_list) * y_true_cnt[i] / len(y_true)
    return shen_f_score


def elsner_local_error(y_true, y_pred, window=3):  # loc3
    match = 0
    total = 0
    anums, gnums = {}, {}
    for i in range(len(y_true)):
        anums[i] = y_true[i]
        gnums[i] = y_pred[i]
        start = 0
        end = max(len(y_true), len(y_pred)) + 1
    for num in range(start, end):
        if num in anums and num in gnums:
            for i in range(-window, 0):
                pos = num + i
                if pos in anums and pos in gnums:
                    if (gnums[pos] == gnums[num]) == (anums[pos] == anums[num]):
                        match += 1
                    total += 1
    return match / total


def clusters_to_contingency(y_true, y_pred):
    # A table, in the form of:
    # https://en.wikipedia.org/wiki/Rand_index#The_contingency_table
    max_x = max(y_true) + 1
    max_y = max(y_pred) + 1
    table = np.zeros((max_x, max_y), dtype=int)
    # print(table)

    for i in range(len(y_true)):
        # print(y_pred[i])
        table[y_true[i]][y_pred[i]] = table[y_true[i]][y_pred[i]] + 1
        # print(table)
    print(table)
    counts_a, counts_g = [], []
    for i in range(max_x):
        counts_a.append(sum(table[i]))

    for j in range(max_y):
        one_j = 0
        for i in range(max_x):
            one_j += table[i][j]
        counts_g.append(one_j)
    return table.tolist(), counts_a, counts_g


def one_to_one(y_true, y_pred):
    contingency, row_sums, col_sums = clusters_to_contingency(y_true, y_pred)
    row_to_num = {}
    col_to_num = {}
    num_to_row = []
    num_to_col = []
    for row_num, row in enumerate(row_sums):
        row_to_num[row] = row_num
        num_to_row.append(row)
    for col_num, col in enumerate(col_sums):
        col_to_num[col] = col_num
        num_to_col.append(col)

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    start_nodes = []
    end_nodes = []
    capacities = []
    costs = []
    source = len(num_to_row) + len(num_to_col)
    sink = len(num_to_row) + len(num_to_col) + 1
    supplies = []
    tasks = min(len(num_to_row), len(num_to_col))
    for row, row_num in row_to_num.items():
        start_nodes.append(source)
        end_nodes.append(row_num)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    for col, col_num in col_to_num.items():
        start_nodes.append(col_num + len(num_to_row))
        end_nodes.append(sink)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    supplies.append(tasks)
    supplies.append(-tasks)
    for row, row_num in row_to_num.items():
        for col, col_num in col_to_num.items():
            cost = 0
            print(row, row_num)
            if col in contingency[row_num]:
                cost = - contingency[row_num][col_num]
            start_nodes.append(row_num)
            end_nodes.append(col_num + len(num_to_row))
            capacities.append(1)
            costs.append(cost)

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow.
    min_cost_flow.Solve()

    # Score.
    total_count = sum(v for v in row_sums)
    overlap = 0
    for arc in range(min_cost_flow.NumArcs()):
        # Can ignore arcs leading out of source or into sink.
        if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(arc) != sink:
            # Arcs in the solution have a flow value of 1. Their start and end nodes
            # give an assignment of worker to task.
            if min_cost_flow.Flow(arc) > 0:
                row_num = min_cost_flow.Tail(arc)
                col_num = min_cost_flow.Head(arc)
                col = num_to_col[col_num - len(num_to_row)]
                row = num_to_row[row_num]
                if col in contingency[row]:
                    overlap += contingency[row][col]
    print("{:5.2f}   one-to-one".format(overlap * 100 / total_count))


def overlap(y_true, y_pred, index_true, index_pred):
    right = 0
    total = 0
    for i in range(len(y_true)):
        if y_true[i] == index_true:
            total += 1
            if y_pred[i] == index_pred:
                right += 1
    return right, total


def one_to_one_me(y_true, y_pred):
    total_num = len(y_true)
    cluster_num = max(y_true)
    truth_list = [i for i in range(cluster_num)]
    pred_list = [i for i in range(max(y_pred))]
    total_right = 0
    for i in truth_list:
        max_right_i = 0
        j_index = 0
        for j in pred_list:
            right, total = overlap(y_true, y_pred, i, j)
            if right >= max_right_i:
                max_right_i = right
                j_index = j
        total_right += total
        pred_list.pop(pred_list.index(j_index))
        print(total_right, total_num)
        print("{:5.2f} one-to-one".format(total_right * 100 / total_num))
    return total_right / total_num


def compare(predicted_labels, truth_labels, metric):
    if metric == 'purity':
        purity_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            purity_scores.append(calculate_purity_scores(y_true, y_pred))
        return sum(purity_scores) / len(purity_scores)
    elif metric == 'NMI':
        NMI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            NMI_scores.append(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
        return sum(NMI_scores) / len(NMI_scores)
    elif metric == 'ARI':
        ARI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            ARI_scores.append(metrics.adjusted_rand_score(y_true, y_pred))
        return sum(ARI_scores) / len(ARI_scores)
    elif metric == "shen_f":
        f_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            f_scores.append(calculate_shen_f_score(y_true, y_pred))
        return sum(f_scores) / len(f_scores)
    elif metric == "Loc3":
        loc_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            loc_scores.append(elsner_local_error(y_true, y_pred))
        return sum(loc_scores) / len(loc_scores)
    elif metric == "one2one":
        loc_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            loc_scores.append(one_to_one_me(y_true, y_pred))
        return sum(loc_scores) / len(loc_scores)
