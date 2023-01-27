from reserved_words import reserved
import string
common_short_names = {"ng", "_2", "x_", "rq", "\\9", "ww", "nn", "bc", "te",
                      "io", "v7", "dm", "m0", "d1", "mr", "x3", "nm", "nu", "jc", "wy", "pa", "mn",
                      "a_", "xz", "qr", "s1", "jo", "sw", "em", "jn", "cj", "j_"}
                      
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

def get_targets(line, users):
    targets = set() # 重复的被删除
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

def read_data(filename, is_test=False): 
    users = set()
    
    text_ascii = [l.strip().split() for l in open(filename)] # 原文 
    for line in text_ascii:
        update_users(line, users)
    for line in text_ascii:
        update_users(line, users)
    info = []

    nexts = {}
    for line_no, line in enumerate(text_ascii):
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

        last_from_user = prev.get(user, None)  # get the utterance from the use in history
        if not system:
            prev[user] = line_no  # wirte the history according to the user
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:  # avoid getting prev
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]  # get the next

        info.append((user, targets))
        total = 0
        withtarget = 0
    for user, target in info:
        if target:
            withtarget += 1
        total += 1
    return total, withtarget



if __name__ == "__main__":
    import os
    train_dirs = "../Disentangle/DSTC8_DATA/Task_4/train/"
    train_paths = os.listdir(train_dirs)
    train_paths = [os.path.join(train_dirs, path) for path in train_paths if path.endswith('.ascii.txt')]
    alltotal = 0
    allwithtarget = 0
    for filename in train_paths:
        total, withtarget = read_data(filename)
        alltotal += total
        allwithtarget += withtarget
        print("Total speaker number is:", total)
        print("With target speaker number is:", withtarget)
    print("All Total speaker number is:", alltotal)
    print("All With target speaker number is:", allwithtarget)

