from bloom_filter import BloomFilter

exact = False

L = [[11, 20], [30, 40], [5, 10], [40, 30], [10, 5]]

sort_pair = sorted

S = set() if exact else BloomFilter(max_elements=10000, error_rate=0.1)
matches = []
for pair in L:
    pair = tuple(sort_pair(pair))
    if pair in S:
        matches.append(pair)
    S.add(pair)

print(matches)

import heapq
import numpy as np

L = list(range(20))
np.random.shuffle(L)

def iter_median(L_iter):
    H_min = []
    H_max = []
    med = 0
    L_dbg = []
    for x in L_iter:
        L_dbg.append(x)
        #
        if x <= med:
            heapq.heappush(H_max, -x)
        else:
            heapq.heappush(H_min, x)
        #
        imbalance = len(H_max) - len(H_min)
        if imbalance > 1:
            x_mv = heapq.heappop(H_max)
            heapq.heappush(H_min, -x_mv)
        if imbalance < -1:
            x_mv = heapq.heappop(H_min)
            heapq.heappush(H_max, -x_mv)
        #
        imbalance = len(H_max) - len(H_min)
        if imbalance == 0:
            med = 0.5 * (H_min[0] - H_max[0])
        elif imbalance == 1:
            med = -H_max[0]
        elif imbalance == -1:
            med = H_min[0]
        else:
            assert False, ":("
        med = float(med)
        #
        print("lower %s" % str(H_max))
        print("upper %s" % str(H_min))
        print("med %f %f" % (med, np.median(L_dbg)))
        assert np.isclose(med, np.median(L_dbg))
    return med


[0, 1, 1]
[0, 0, 1, 1, 2, 3, 3, 4, 4]

def find_index(L):
    print("L %s" % L)
    N = len(L)
    # ii could be in [0, N-1], but will always be even index for solution
    lower = 0
    upper = N - 1
    assert upper % 2 == 0, 'upper even'
    while lower < upper:
        mid = (lower + upper) // 2  # floordiv, but always even anyway
        mid = mid - (mid % 2)
        print(lower, mid, upper)
        assert lower <= mid, 'lower range'
        assert mid <= upper, 'upper range'
        assert mid % 2 == 0, 'mid even'
        #
        if L[mid] != L[mid + 1]:
            # then solution <= mid
            upper = mid
        else:
            # then solution > mid
            lower = mid + 2
    assert lower == upper, 'closed'
    return L[lower]


@given(lists(integers(-100, 100), unique=True, min_size=1))
def test_find_index(L):
    solution = L[0]
    L = sorted(L)
    L = sum(([x, x] for x in L), [])
    L.remove(solution)
    #
    y = find_index(L)
    assert y == solution


def find_dominant_index(L):
    if len(L) < 2:
        return -1
    # init
    if L[0] <= L[1]:
        gold = 1
        silver = 0
    else:
        gold = 0
        silver = 1
    # Find top-2
    for ii, xx in enumerate(L[2:], 2):
        if xx >= L[silver]:  # Could save in var if helps
            if xx >= L[gold]:
                silver = gold
                gold = ii
            else:
                silver = ii
    # check
    if L[gold] >= 2 * L[silver]:
        return gold
    # or -1
    return -1


@given(lists(integers(0, 99), min_size=2, max_size=50))
def test_find_dominant_index(L):
    ii = find_dominant_index(L)
    print(L, ii)
    #
    sorted_L = sorted(L)
    if sorted_L[-1] >= 2 * sorted_L[-2]:
        assert L[ii] == sorted_L[-1]
    else:
        assert ii == -1


def find_repeat(L):
    N = len(L)
    for ii in range(1, N):  # Could stop before N
        if N % ii == 0:  # otherwise we know it can't be repeat
            rr = N // ii
            if L == L[:ii] * rr:
                return True, (ii, rr)
    return False, (0, 0)


@given(lists(integers(0, 99), min_size=1, max_size=50), integers(2, 20))
def test_rep(L, rep):
    L2 = L * rep
    assert find_repeat(L2)[0]


def find_sum(L, target):
    D = {}
    for ii, x in enumerate(L):
        if x in D:
            return D[x], ii
        else:
            D[target - x] = ii
    assert False


def least_index_sum(L1, L2):
    D1 = {ss: ii for ii, ss in enumerate(L1)}
    best_el = None
    best_score = len(L1) + len(L2)  # upperbound can always beat
    for ii, ss in enumerate(L2):
        if (ss in D1) and (ii + D1[ss] < best_score):
            best_score = ii + D1[ss]
            best_el = ss
    return best_el


def least_index_sum_2(L1, L2):
    S = set(L1) & set(L2)
    D1 = {ss: ii for ii, ss in enumerate(L1)}
    D2 = {ss: ii for ii, ss in enumerate(L2)}
    DS = {ss: D1[ss] + D2[ss] for ss in S}
    key_min = None if len(DS) == 0 else min(DS.keys(), key=(lambda k: DS[k]))
    return key_min


@given(lists(integers(0, 10)), lists(integers(0, 10)))
def test_min(L1, L2):
    R1 = least_index_sum(L1, L2)
    R2 = least_index_sum(L1, L2)
    assert R1 == R2


from collections import deque


class RecentCounter(object):
    def __init__(self):
        self.q = deque()
    #
    def ping(self, t):
        self.q.append(t)
        #
        expire_time = t - 3000
        #
        t_ = None
        while (t_ is None) or (t_ < expire_time):
            t_ = self.q.popleft()
        assert t_ is not None
        assert t_ >= expire_time
        self.q.appendleft(t_)  # put back last
        print(list(self.q))
        return len(self.q)




def segmenter(L, D):
    if len(L) == 0:  # base case
        return []
    #
    max_try = len(L) + 1  # Could make this shorter based on lengths in D
    best_sol = None
    best_len = len(L) + 10  # some upper bound
    for ii in range(1, max_try):
        if L[:ii] in D:  # could use continue to avoid nesting
            sol = segmenter(L[ii:], D)
            if (sol is not None) and len(sol) < best_len:
                best_sol = [L[:ii]] + sol
                best_len = len(sol)
                print(best_sol, best_len)
    return best_sol





def segmenter2(L, D, _len_list=None):
    if len(L) == 0:  # base case
        return []
    #
    if _len_list is None:
        _len_list = sorted(set(len(ss) for ss in D))
        assert all(ss > 0 for ss in _len_list)
    #
    best_sol = None
    best_len = len(L) + 10  # some upper bound
    # or could do greedy approach of starting with longest and going smaller if doesn't fit
    for ii in _len_list:
        if L[:ii] in D:  # could use continue to avoid nesting
            sol = segmenter2(L[ii:], D, _len_list=_len_list)
            if (sol is not None) and len(sol) < best_len:
                best_sol = [L[:ii]] + sol
                best_len = len(sol)
    return best_sol



def segmenter_greedy(L, D, _len_list=None):
    if len(L) == 0:  # base case
        return []
    #
    if _len_list is None:
        _len_list = sorted(set(len(ss) for ss in D))[::-1]
        assert all(ss > 0 for ss in _len_list)
    #
    for ii in _len_list:
        if L[:ii] in D:  # could use continue to avoid nesting
            sol = segmenter_greedy(L[ii:], D, _len_list=_len_list)
            if sol is not None:
                return [L[:ii]] + sol
    return None




@given(lists(lists(integers(0, 9), min_size=1).map(tuple), min_size=1, unique=True).map(set), lists(integers()))
def test_segmenter(D, idx):
    DS = sorted(D)
    L = sum([DS[ii % len(DS)] for ii in idx], ())
    print('<', L, D)
    #
    sol = segmenter_greedy(L, D)
    print('>', L, sol)
    assert sum(sol, ()) == L
    assert len(sol) <= len(L)



