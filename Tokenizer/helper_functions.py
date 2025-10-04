def get_stats(ids, count=None):
    count = {} if count is None else count
    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count

def merge(ids, pair, idx):
    i = 0
    new_ids = []

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids