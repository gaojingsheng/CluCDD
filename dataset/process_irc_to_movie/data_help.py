# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tokenization import *
import string
import numpy as np
import os

FEATURES = 77
from reserved_words import reserved


def update_user(users, user):
    if user in reserved:
        return
    all_digit = True
    for char in user:
        if char not in string.digits:
            all_digit = False
    if all_digit:
        return
    users.add(user.lower())


def update_users(line, users):
    if len(line) < 2:
        return
    user = line[1]
    if user in ["Topic", "Signoff", "Signon", "Total", "#ubuntu"
                                                       "Window", "Server:", "Screen:", "Geometry", "CO,",
                "Current", "Query", "Prompt:", "Second", "Split",
                "Logging", "Logfile", "Notification", "Hold", "Window",
                "Lastlog", "Notify", 'netjoined:']:
        # Ignore as these are channel commands
        pass
    else:
        if line[0].endswith("==="):
            parts = ' '.join(line).split("is now known as")
            if len(parts) == 2 and line[-1] == parts[-1].strip():
                user = line[-1]
        elif line[0][-1] == ']':
            if user[0] == '<':
                user = user[1:]
            if user[-1] == '>':
                user = user[:-1]

        user = user.lower()
        update_user(users, user)
        # This is for cases like a user named |blah| who is
        # refered to as simply blah
        core = [char for char in user]
        while len(core) > 0 and core[0] in string.punctuation:
            core.pop(0)
        while len(core) > 0 and core[-1] in string.punctuation:
            core.pop()
        core = ''.join(core)
        update_user(users, core)


# Names two letters or less that occur more than 500 times in the data
common_short_names = {"ng", "_2", "x_", "rq", "\\9", "ww", "nn", "bc", "te",
                      "io", "v7", "dm", "m0", "d1", "mr", "x3", "nm", "nu", "jc", "wy", "pa", "mn",
                      "a_", "xz", "qr", "s1", "jo", "sw", "em", "jn", "cj", "j_"}


def get_targets(line, users):
    targets = set() 
    for token in line[2:]:
        token = token.lower()
        user = None
        if token in users and len(token) > 2:
            user = token
        else:
            core = [char for char in token]
            while len(core) > 0 and core[-1] in string.punctuation:
                core.pop()
                nword = ''.join(core)
                if nword in users and (len(core) > 2 or nword in common_short_names):
                    user = nword
                    break
            if user is None:
                while len(core) > 0 and core[0] in string.punctuation:
                    core.pop(0)
                    nword = ''.join(core)
                    if nword in users and (len(core) > 2 or nword in common_short_names):
                        user = nword
                        break
        if user is not None:
            targets.add(user)
    return targets


def lines_to_info(text_ascii):
    users = set()
    for line in text_ascii:
        update_users(line, users)
    chour = 12
    cmin = 0
    info = []
    target_info = {}
    nexts = {}
    for line_no, line in enumerate(text_ascii):# 
        if line[0].startswith("["):
            user = line[1][1:-1]
            nexts.setdefault(user, []).append(line_no)
    prev = {}
    for line_no, line in enumerate(text_ascii):
        user = line[1]
        system = True
        if line[0].startswith("["):
            chour = int(line[0][1:3])
            cmin = int(line[0][4:6])
            user = user[1:-1]
            system = False
        is_bot = (user == 'ubottu' or user == 'ubotu')
        targets = get_targets(line, users)
        for target in targets:
            target_info.setdefault((user, target), []).append(line_no)
        last_from_user = prev.get(user, None)  # get the utterance from the use in history
        if not system:
            prev[user] = line_no  # wirte the history according to the user
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:  # avoid getting prev
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]  # get the next

        info.append((user, targets, chour, cmin, system, is_bot, last_from_user, line, next_from_user))
        # print(user)# lizeyu
    return info, target_info


def get_time_diff(info, a, b):
    if a is None or b is None:
        return -1
    if a > b:
        t = a
        a = b
        b = t
    ahour = info[a][2]
    amin = info[a][3]
    bhour = info[b][2]
    bmin = info[b][3]
    if ahour == bhour:
        return bmin - amin
    if bhour < ahour:
        bhour += 24
    return (60 - amin) + bmin + 60 * (bhour - ahour - 1)


cache = {}

def get_features(name, query_no, link_no, text_ascii, text_tok, info, target_info, do_cache=True):
    global cache
    if (name, query_no, link_no) in cache:
        return cache[name, query_no, link_no]

    features = []

    quser, qtargets, qhour, qmin, qsystem, qis_bot, qlast_from_user, qline, qnext_from_user = info[query_no]
    luser, ltargets, lhour, lmin, lsystem, lis_bot, llast_from_user, lline, lnext_from_user = info[link_no]

    # General information about this sample of data
    # Year
    for i in range(2004, 2018):
        features.append(str(i) in name)
    # Number of messages per minute
    start = None
    end = None
    for i in range(len(text_ascii)):
        if start is None and text_ascii[i][0].startswith("["):
            start = i
        if end is None and i > 0 and text_ascii[-i][0].startswith("["):
            end = len(text_ascii) - i - 1
        if start is not None and end is not None:
            break
    diff = get_time_diff(info, start, end)
    msg_per_min = len(text_ascii) / max(1, diff)
    cutoffs = [-1, 1, 3, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= msg_per_min < end)

    # Query
    #  - Normal message or system message
    features.append(qsystem)
    #  - Hour of day
    features.append(qhour / 24)
    #  - Is it targeted
    features.append(len(qtargets) > 0)
    #  - Is there a previous message from this user?
    features.append(qlast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if qlast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[qlast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if qlast_from_user is None else query_no - qlast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, query_no, qlast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(qis_bot)

    ###- Often a person asks a question as the first thing they do after entering the channel. Could be a useful feature. Encode as how long ago they entered, or if this is their first message since entering.

    # Link
    #  - Normal message or system message
    features.append(lsystem)
    #  - Hour of day
    features.append(lhour / 24)
    #  - Is it targeted
    features.append(link_no != query_no and len(ltargets) > 0)
    #  - Is there a previous message from this user?
    features.append(link_no != query_no and llast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if link_no == query_no or llast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[llast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if llast_from_user is None else link_no - llast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(link_no != query_no and start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, link_no, llast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(lis_bot)
    #  - Is the message after from the same user?
    features.append(link_no != query_no and link_no + 1 < len(info) and luser == info[link_no + 1][0])
    #  - Is the message before from the same user?
    features.append(link_no != query_no and link_no - 1 > 0 and luser == info[link_no - 1][0])

    # Both
    #  - Is this a self-link?
    features.append(link_no == query_no)
    #  - How far apart in messages are the two?
    dist = query_no - link_no
    features.append(min(100, dist) / 100)
    features.append(dist > 1)
    #  - How far apart in time are the two?
    time = get_time_diff(info, link_no, query_no)
    features.append(min(100, time) / 100)
    cutoffs = [-1, 0, 1, 5, 60, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Does the link target the query user?
    features.append(quser.lower() in ltargets)
    #  - Does the query target the link user?
    features.append(luser.lower() in qtargets)
    #  - none in between from src?
    features.append(link_no != query_no and (qlast_from_user is None or qlast_from_user < link_no))
    #  - none in between from target?
    features.append(link_no != query_no and (lnext_from_user is None or lnext_from_user > query_no))
    #  - previously src addressed target?
    #  - future src addressed target?
    #  - src addressed target in between?
    if link_no != query_no and (quser, luser) in target_info:
        features.append(min(target_info[quser, luser]) < link_no)
        features.append(max(target_info[quser, luser]) > query_no)
        between = False
        for num in target_info[quser, luser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - previously target addressed src?
    #  - future target addressed src?
    #  - target addressed src in between?
    if link_no != query_no and (luser, quser) in target_info:
        features.append(min(target_info[luser, quser]) < link_no)
        features.append(max(target_info[luser, quser]) > query_no)
        between = False
        for num in target_info[luser, quser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - are they the same speaker?
    features.append(luser == quser)
    #  - do they have the same target?
    features.append(link_no != query_no and len(ltargets.intersection(qtargets)) > 0)
    #  - Do they have words in common?
    ltokens = set(text_ascii[link_no])
    qtokens = set(text_ascii[query_no])
    common = len(ltokens.intersection(qtokens))
    if link_no != query_no and len(ltokens) > 0 and len(qtokens) > 0:
        features.append(common / len(ltokens))
        features.append(common / len(qtokens))
    else:
        features.append(False)
        features.append(False)
    features.append(link_no != query_no and common == 0)
    features.append(link_no != query_no and common == 1)
    features.append(link_no != query_no and common > 1)
    features.append(link_no != query_no and common > 5)

    # Convert to 0/1
    final_features = []
    for feature in features:
        if feature == True:
            final_features.append(1.0)
        elif feature == False:
            final_features.append(0.0)
        else:
            final_features.append(feature)
    ###    print(query_no, link_no, ' '.join([str(int(v)) for v in final_features]))

    if do_cache:
        cache[name, query_no, link_no] = final_features
    return final_features

def process_to_movie(filenames, is_test=False): 
    instances = []
    cluster_dic = {} 
    cluster_index = 0
    for filename in filenames:
        if filename not in cluster_dic:
            cluster_dic[filename] = {}
        name = filename 
        for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
            if filename.endswith(ending):
                name = filename[:-len(ending)]
        text_ascii = [l.strip().split() for l in open(name + ".ascii.txt")] # 原文 

        text_tok = []
        for l in open(name + ".tok.txt"):
            l = l.strip().split()
            if len(l) > 0 and l[-1] == "</s>":# remove </s>
                l = l[:-1]
            if len(l) == 0 or l[0] != '<s>':# if there is nothing in it, insert <s> in the begining
                l.insert(0, "<s>")
            text_tok.append(l)# tokn
        # print(text_tok)
        info, target_info = lines_to_info(text_ascii)
        # print(target_info)
        links = {}
        if is_test:
            for i in range(1000, min(2000, len(text_ascii))):
                links[i] = []
        else: # train
            for line in open(name + ".annotation.txt"):
                nums = [int(v) for v in line.strip().split() if v != '-'] # ['1498', '1499', '-']-> ['1498', '1499']
                links.setdefault(max(nums), []).append(min(nums))
        for link, nums in links.items():
            
            instances.append((name + ".annotation.txt", link, nums, text_ascii, text_tok, info, target_info))
            link_arr = [link]
            link_arr.extend(nums)
            flag = True
            temp = cluster_index
            for num in link_arr:
                if num in cluster_dic[filename]:
                    flag = False
                    temp = cluster_dic[filename][num]
            for num in link_arr:
                cluster_dic[filename][num] = temp
            if flag:
                cluster_index +=1

        # json_str = json.dumps(cluster_dic)
        # with open("../Disentangle/scripts/logs/record_test.json","w") as f:
        #     json.dump(json_str,f)
    return instances, cluster_dic

def read_data(filenames, is_test=False): 
    instances = []
    cluster_dic = {} 
    cluster_index = 0
    for filename in filenames:
        if filename not in cluster_dic:
            cluster_dic[filename] = {}
        name = filename 
        for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
            if filename.endswith(ending):
                name = filename[:-len(ending)]
        text_ascii = [l.strip().split() for l in open(name + ".ascii.txt")] # 原文 
        text_tok = []
        for l in open(name + ".tok.txt"):
            l = l.strip().split()
            if len(l) > 0 and l[-1] == "</s>":
                l = l[:-1]
            if len(l) == 0 or l[0] != '<s>':
                l.insert(0, "<s>")
            text_tok.append(l)

        info, target_info = lines_to_info(text_ascii)
        links = {}
        if is_test:
            for i in range(1000, min(2000, len(text_ascii))):
                links[i] = []
        else: # train
            for line in open(name + ".annotation.txt"):
                nums = [int(v) for v in line.strip().split() if v != '-'] 
                links.setdefault(max(nums), []).append(min(nums)) 
        for link, nums in links.items():
            instances.append((name + ".annotation.txt", link, nums, text_ascii, text_tok, info, target_info))
            link_arr = [link]
            link_arr.extend(nums)
            flag = True
            temp = cluster_index
            for num in link_arr:
                if num in cluster_dic[filename]:
                    flag = False
                    temp = cluster_dic[filename][num]

            for num in link_arr:
                cluster_dic[filename][num] = temp
            if flag:
                cluster_index +=1

    return instances, cluster_dic


def remove_time(line):
    items = line.strip().split(' ')
    if items[0].startswith('[') and items[0].endswith(']'):
        return ' '.join(items[1:])
    else:
        return ' '.join(items)


def load_data(instances, max_turns):
    dataset = []
    data_query = []
    data_name = []
    for instance in instances:
        name, query, gold, text_ascii, text_tok, info, target_info = instance
        
        gold = [v for v in gold if v > query - max_turns]
        candi = []
        label = []
        features = []
        query_list = []
        name_list = []
        target_index = max_turns - 1 # 50 cluster
        candi_indexs = range(max_turns) # max_turn
        query_text = remove_time(' '.join(text_ascii[query]).strip())

        for i in range(query - max_turns + 1, query + 1):
            candi.append((query_text, remove_time(' '.join(text_ascii[i]).strip())))
            if i not in gold:
                label.append(0.0)
            else:
                label.append(1.0 / len(gold))
            feature = get_features(name, query, i, text_ascii, text_tok, info, target_info)
            features.append(feature)
            query_list.append(i) 
            name_list.append(name)
        gold = [max_turns - (query - v + 1) for v in gold]
        dataset.append((candi, target_index, candi_indexs, label, features, gold))
        data_query.append(query_list)
        data_name.append(name_list)
    return dataset, data_query, data_name


def compute_cluster(data_query, data_name, cluster_dic):
    cluster = []
    count_bad = 0
    jquery = {}
    for query, name in zip(data_query,data_name):
        cluster_list = []
        jquery_small = {}
        for index in range(len(query)):
            if query[index] not in cluster_dic[name[index]]:
                cluster_list.append(-1)
                jquery_small[query[index]] = -1

            else:
                cluster_list.append(cluster_dic[name[index]][query[index]])
                # print(name[index],'-----------------',cluster_list)
                jquery_small[query[index]] = cluster_dic[name[index]][query[index]]
       
        cluster.append(cluster_list)
        
    # json_str = json.dumps(jquery)
    #     # print(cluster_dic)
    # with open("/home/gaojingsheng/project/NLP/Disentangle/scripts/logs/cluster_dev.json","w") as f:
    #     json.dump(json_str,f)
    return cluster

def convert_index(cluster): 
    after_convert=[]
    for sample_cluster in cluster: 
        index = 1
        dic = {}
        convert_buffer = []
        dic[sample_cluster[-1]] = 0
        for sample in sample_cluster: 
            if sample ==-1:
                convert_buffer.append(-1)
                continue
            if sample not in dic:
                dic[sample] = index
                convert_buffer.append(dic[sample])
                index +=1
            else:
                convert_buffer.append(dic[sample])
        if max(convert_buffer)==0:
            if -1 not in convert_buffer:
                convert_buffer[0] = index
            for i in range(len(convert_buffer)):
                if convert_buffer[i]!=0:
                    convert_buffer[i] = index
                    break

        after_convert.append(convert_buffer)
    return after_convert



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def sent2idxs(text, tokenizer, max_seq_length):
    text_a, text_b = text 
    tokens = []
    segment_ids = []
    tokens_a = tokenizer.tokenize(text_a)
    tokens_a = [token for token in tokens_a if token.strip() != '']
    tokens_b = tokenizer.tokenize(text_b)
    tokens_b = [token for token in tokens_b if token.strip() != '']
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)  # the mask of tokens

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


def batch2idxs(data,data_cluster,  max_turns, tokenizer, max_seq_length):
    batch_input_ids = []
    batch_input_masks = []
    batch_token_types = []
    batch_masks = []
    batch_labels = []
    batch_target_index = []
    batch_candi_indexs = []
    batch_features = []
    batch_gold = []
    for line in data:
        candi, target_index, candi_indexs, label, features, gold = line
        tmp_input_ids = []
        tmp_input_masks = []
        tmp_token_types = []
        for text in candi:
            input_ids, input_mask, segment_ids = sent2idxs(text, tokenizer, max_seq_length) # 两句话
            tmp_input_ids.append(input_ids)
            tmp_input_masks.append(input_mask)
            tmp_token_types.append(segment_ids)
        batch_input_ids.append(tmp_input_ids)
        batch_input_masks.append(tmp_input_masks)
        batch_token_types.append(tmp_token_types)
        batch_masks.append([1] * max_turns) 
        batch_labels.append(label) 
        batch_target_index.append(target_index)
        batch_candi_indexs.append(candi_indexs)
        batch_features.append(features)
        batch_gold.append(gold)
    return np.array(batch_input_ids, 'int32'), np.array(batch_input_masks, 'int32'), np.array(batch_token_types,
                                                                                              'int32'), np.array(
        batch_masks, 'int32'), \
           np.array(batch_labels, 'float32'), np.array(batch_target_index, 'int32').reshape(-1, 1), np.array(
        batch_candi_indexs, 'int32'), np.array(batch_features, 'float32'), batch_gold, np.array(data_cluster,'int32')


if __name__ == "__main__":

    path = "../DSTC8_DATA/Task_4/train/"
    train_paths = os.listdir(path)
    train_paths = [os.path.join(path, path) for path in train_paths if path.endswith('.annotation.txt')]
    process_to_movie(train_paths, is_test=False)