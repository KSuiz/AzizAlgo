import math
from collections import Counter, deque
from copy import deepcopy
from fractions import Fraction

# Turn a utility profile of size n x m into an ordinal profile
# of size n x cn, where c = ceil(m/n)
def generate_prefs(utils):
    n = len(utils)
    m = len(utils[0])
    # Find the minimum utility so dummy items are always least preferred
    min_util = min([min(l) for l in utils])
    c = int(math.ceil(float(m) / n))
    t_obj = c * n
    # Make the dummy items
    dummy = [min_util - 1] * (t_obj - m)
    # Return n, m, c, and the ordinal profile
    # If object i and j have the same utility, i is prefered over j iff i < j
    return n, m, c, [list(zip(*sorted(zip(l + dummy, range(t_obj)), key = lambda v: (-v[0], v[1])))[1]) for l in utils]

# Run Probabilistic Serial on given preferences
def run_ps(n, c, const_prefs):
    # We need to modify this, so copy it
    prefs = deepcopy(const_prefs)
    # Begin with all objects having a unit amount remaining
    objs = [1] * c * n
    # Begin with each agent having eaten nothing
    alloc = [[0] * c * n for i in range(n)]
    # Loop until we have no preferences remaining
    # The object have been padded, so if agent 1 has none left, no one else does
    while len(prefs[0]):
        # Find the max amount everyone can eat before exhausting their most prefered item
        counts = Counter([l[0] for l in prefs]).most_common()
        to_eat = 1
        for count in counts:
            # The most we can eat is the amount of the item left / # agents who want it
            amt = Fraction(objs[count[0]]) / count[1]
            if to_eat > amt:
                to_eat = amt
        # For each agent, eat 1/split of their most preferred item
        for agent in range(n):
            fav_obj = prefs[agent][0]
            # Add the amount to the agent's allocation, and remove it from the item
            alloc[agent][fav_obj] += to_eat
            objs[fav_obj] -= to_eat
        # Remove the preferences for items with nothing remaining
        prefs = [[x for x in y if not objs[x] == 0] for y in prefs]
    return alloc
        
# Break an n x cn allocation into a cn x cn allocation by adding representatives
def representative(prefs, alloc, n, c):
    res_alloc = []
    for agent in range(n):
        for rep in range(c):
            # Begin with an empty allocation with 1 unit to eat
            suballoc = [0] * (c * n)
            amt = 1
            # Continue eating agent's allocation until 1 unit is used
            while amt > 0:
                # Get the highest preference item and its allocated amount
                cur_obj = prefs[agent][0]
                val = alloc[agent][cur_obj]
                # Eat only up to the remaining amount, removing the item if we eat it all
                if amt < val:
                    val = amt
                else:
                    prefs[agent].pop(0)
                # Add the amount to the representative's allocation, and remove it from
                # the overall allocation, as well as the amount left to eat
                suballoc[cur_obj] = val
                alloc[agent][cur_obj] -= val
                amt -= val
            # Add the representative allocation to our current allocation
            # 0 ... c - 1 are agent 1's representatives,
            # c ... 2c - 1 are agent 2's representatives, etc.
            res_alloc.append(suballoc)
    return res_alloc

# Used in augment to check if two nodes are connected
def connected(alloc, u, v, n):
    if u < n:
        return alloc[u][v - n] > 0
    return alloc[v][u - n] > 0

# Used in augment to check if a path jump is allowed
# Specifically:
#   If moving from left to right, not a matched edge
#   If moving from right to left, a matched edge
def allowable(match, u, v, n):
    if u < n:
        return not match[u] == v
    return match[v] == u

# Find an augmenting path in a bipartite partially-matched graph
def augment(alloc, match, n):
    found = [False] * (2 * n)
    pred = [-1] * (2 * n)
    end = -1
    q = deque()
    # Our BFS "starts" at all unmatched left vertices
    for i in range(n):
        if match[i] == -1:
            found[i] = True
            q.append(i)
    # Standard BFS to find a path
    while q:
        v = q.popleft()
        # Once we have an unmatched right vertex, we're done
        if v >= n and match[v] == -1:
            end = v
            break
        # Loop through all vertices we can connect to
        base = 0 if v >= n else n
        for i in range(base, base + n):
            # Skip disconnected edges, already found vertices, and
            # edges we can't cross (see allowable)
            if not connected(alloc, v, i, n):
                continue
            if not allowable(match, v, i, n):
                continue
            if found[i]:
                continue
            q.append(i)
            found[i] = True
            # Save the predecessor to re-make the path
            pred[i] = v
    # Check if we didn't find an augmenting path
    if end == -1:
        return None
    path = deque()
    # Reconstruct the path
    while not end == -1:
        path.appendleft(end)
        end = pred[end]
    return list(path)

def hopcroft_karp(alloc, n):
    # The left side (agents) are 0 ... n - 1
    # The right side (objects) are n ... 2n - 1
    # Begin with no matching
    match = [-1] * (2 * n)
    while 1:
        # Find an augmenting path, if it exists
        path = augment(alloc, match, n)
        if path == None:
            break
        # Apply the found path by changing the matchings
        for i in range(len(path)):
            if path[i] < n:
                match[path[i]] = path[i + 1]
            else:
                match[path[i]] = path[i - 1]
    # n ... 2n - 1 are only for bookkeeping, so drop them
    # Objects start at n, so shift them back, at the same time
    return [x - n for x in match[0:n]]

# Reduce a cn x m allocation to an n x m one, collapsing representatives
def reduce(perm, n, m, c):
    k = 0
    reduced = []
    for agent in range(n):
        sum_arr = [0] * m
        for rep in range(c):
            # Add the representatives' allocations together
            sum_arr = map(lambda a, b: a + b, sum_arr, perm[k])
            k += 1
        reduced.append(sum_arr)
    return reduced

# Decompose an allocation into a lottery over discrete allocations
def decompose(alloc, n, c, m):
    # Number of non-zero entries
    non_zero = n * n * c * c
    decomposition = []
    # Count all entries that are already zero
    for agent in alloc:
        for val in agent:
            if val == 0:
                non_zero -= 1
    # Keep decomposing until the array is empty
    while non_zero:
        # Find a perfect matching
        match = hopcroft_karp(alloc, n * c)
        # Find the minimum value in the allocation over the matching
        min_val = alloc[0][match[0]]
        for i in range(1, n * c):
            if min_val > alloc[i][match[i]]:
                min_val = alloc[i][match[i]]
        # Create our discrete allocation, ignoring dummy items
        # Here, i is the agent/representative, and j is the item
        disc_alloc = [[0] * m for k in range(n * c)]
        for i in range(n * c):
            j = match[i]
            # Only add the entry to the permutation if it's not a dummy
            if j < m:
                disc_alloc[i][j] = 1
            # Remove the value from our allocation matrix, and count non-zero entries
            alloc[i][j] -= min_val
            if alloc[i][j] == 0:
                non_zero -= 1
        # Add the reduced matrix (collapsing representatives)
        decomposition.append((min_val, reduce(disc_alloc, n, m, c)))
    return decomposition

# Real dodgy printing function for the decomposition
def printd(dec):
    for part in dec:
        top = str(part[0].numerator)
        bot = str(part[0].denominator)
        height = len(part[1])
        mid = height // 2
        offset = max(len(top), len(bot)) + 1
        for i in range(height):
            strv = ""
            if i == mid - 1:
                strv += top + (" " * (offset - len(top)))
            elif i == mid:
                strv += bot + (" " * (offset - len(bot)))
            else:
                strv += " " * offset
            strv += "["
            strv += " ".join(str(x) for x in part[1][i])
            strv += "]"
            if i == mid:
                strv += " +"
            print(strv)
        print("")

utils = [[4, 3, 2, 1], [4, 2, 3, 1], [1, 2, 3, 4]]
# Get our preference profile
n, m, c, prefs = generate_prefs(utils)
# Get our PS allocation
ps_alloc = run_ps(n, c, prefs)
# Introduce representatives
rep_alloc = representative(prefs, ps_alloc, n, c)
# Decompose into discrete allocations
decomposition = decompose(rep_alloc, n, c, m)
printd(decomposition)
