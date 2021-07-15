'''
Haosen He 2020

This file contains the impmlementations of the Crop-Raiding Model (CRM), the
Random Walk models (RW), and the Myopic Walk (MW) models. Note that these algorithms
produce walks without consuming the resources in the actual map. This facilitates
tuning and model comparison as well as the future implementation of the animal
class.
'''
import itertools as itt
from collections import defaultdict
import random


def RTN(loc, step):
    '''
    Return the round-trip neighboorhood (RTN) given location and steps allowed

    loc -- a 2-tuple indicating a vertex on the lattice
    step -- the number of steps allowed for the agent
    '''
    if step <= 1:
        return {loc}
    l = step//2
    N = set()
    r = loc[0]
    c = loc[1]
    #fixpt = (r-l,c-l)
    for i in range(0, 2*l+1):
        for j in range(0, 2*l+1):
            N.add((r-l+i, c-l+j))
    return N


def STN(loc, step):
    '''
    Return the single-trip neighborhood (STN) given location and number of steps allowed

    loc -- a 2-tuple indicating a vertex on the lattice
    step -- the number of steps allowed for the agent
    '''
    if step == 0:
        return {loc}
    N = set()
    r = loc[0]
    c = loc[1]
    for i in range(0, 2*step+1):
        for j in range(0, 2*step+1):
            N.add((r-step+i, c-step+j))
    return N


def dist(loc1, loc2):
    '''
    This is the D-find sub-algorithm that returns the shortest distance between
    two location tuples.

    loc1, loc2 —- two locations

    Return the shortest distance between the two locations on the lattice
    '''
    r1 = loc1[0]
    r2 = loc2[0]
    c1 = loc1[1]
    c2 = loc2[1]
    return max(abs(r1-r2), abs(c1-c2))


def is_valid(subwalk):
    '''
    Test if a subwalk is valid.

    subwalk —- the subwalk to be tested

    Return True if this subwalk is valid,
    Otherwise return False
    '''
    for i in range(len(subwalk)-1):
        if dist(tuple[i], tuple[i+1]) > 1:
            return False
    else:
        return True


def valid_pair(a, b):
    '''
    Check if the pair of locations have a shortest distance <= 1 (adjacent)

    A, B —- location tuples

    Return True if the pair are adjacent, otherwise return False.
    '''
    if dist(a, b) <= 1:
        return True
    return False


def SC_product(*args, repeat=1):
    '''
    A special Cartesian product that omits invalid sub-walks in the output.

    args —- subwalks to be used as input of the set of all possible walks

    Return the special Cartesian product as a list.
    '''
    pools = [tuple(pool) for pool in args] * repeat
    result = [[pools[0][0]]]
    for pool in pools[1:]:
        result = [x+[y] for x in result for y in pool if valid_pair(x[-1], y)]
    for prod in result:
        yield tuple(prod)


def pos_visit(loc1, loc2, step, repset):
    '''
    This is the P-visit sub-algorithm that finds all possible
    valid walks from loc1 to loc2 without element in repset by
    listing the sets containing the possible location tuples of
    each steps. The list is in sequential order but does not
    include loc2. We put all possible locations into a generator
    to avoid memory leak.

    loc1, loc2 -- 2-tuples representing blocks on lattice
    step -- step length
    repset -- locations to avoid

    Return a set of all possible valid subwalks. If there's no
    such walk, return False.
    '''
    if loc1 == loc2 and step == 0:
        return set()
    store = [{loc1}]
    for i in range(1, step):
        if STN(loc1, i).intersection(STN(loc2, step-i))-repset == set():
            return False
        store.append(STN(loc1, i).intersection(STN(loc2, step-i))-repset)
    output = set()
    for i in SC_product(*store):
        output.add(i)
    return output


def TCDutil(walk, crop_amount, tree_amount, trespass_cost, mg_tree, mg_crop, mg_water, alpha, beta, moving_cost):
    '''
    This is the Cobb-Douglas utility function.

    walk -- a list of 2-tuples representing agent's walk
    crop_amount -- a dictionary that keeps track of the amount of food in each crop block
    tree_amount -- a dictionary that keeps track of the amount of food in each tree block
    trespass_cost, mg_tree, mg_crop, mg_water, alpha, beta, moving_cost -- parameters

    Return the utility gain from the walk.
    '''
    food = 0
    water = 0
    cost = 0
    if len(walk) <= 1:  # if stay in the habitat
        return 0
    else:
        count = 0
        for i in walk:
            if i in cropset:
                count += 1
            elif i not in cropset:
                cost += trespass_cost*(count)
                count = 0
        fq = defaultdict(int)  # frequency of a block
        fq[walk[0]] += 1  # initial block
        nomoves = 0  # times the ele does not move
        for i in range(1, len(walk)):
            fq[walk[i]] += 1
            if walk[i] == walk[i-1]:  # does not move
                nomoves += 1
        for j in fq.keys():
            if j in treeset:  # calculate food increment from trees
                if tree_amount[j] >= int(fq[j]*mg_tree):
                    food += fq[j]*mg_tree
                else:
                    food += tree_amount[j]
            elif j in cropset:  # food increment from crops
                if crop_amount[j] >= int(fq[j]*mg_crop):
                    food += fq[j]*mg_crop
                else:
                    food += crop_amount[j]
            elif j in waterset:  # water increment form water
                water += fq[j]*mg_water
        cost = cost + (len(walk)-nomoves)*moving_cost
        # length of the walk need to +1 for the last walk into the habitat and -1 for the
        # initial location, so no change
        return (food**alpha)*(water**beta) - cost


def Tsearch_walk(crop_amount, tree_amount, step, trespass_cost, mg_tree, mg_crop, mg_water, habitat, alpha, beta, moving_cost):
    '''
    This is the CRM search algorithm (core algorithm).

    crop_amount -- a dictionary that keeps track of the amount of food in each crop block
    tree_amount -- a dictionary that keeps track of the amount of food in each tree block
    habitat -- habitat of the agent
    step -- max steps allowed
    trespass_cost, mg_tree, mg_crop, mg_water, alpha, beta, moving_cost -- parameters. beta = 1-alpha when tuning.

    Return the set of optimal walks.
    '''
    rtn = RTN(habitat, step)
    food = {x for x in cropset.union(treeset) if x in rtn}  # prune infeasible food
    water = {y for y in waterset if y in rtn}  # prune infeasible water
    maxset = {(habitat,)*(step+1)}  # set of optimal walks
    maxUtil = 0  # current max utility
    searched = {f: set() for f in food}
    for i in itt.product(food, water):  # 'itt.product' generates the Cartesian product of the two sets
        if dist(habitat, i[0]) + dist(i[0], i[1]) + dist(i[1], habitat) <= step:  # feasibility check
            for k in range(dist(habitat, i[0]), step - dist(i[1], habitat) - dist(i[0], i[1]) + 1):
                for j in range(dist(i[0], i[1]), step - k - dist(i[1], habitat) + 1):
                    A = pos_visit(habitat, i[0], k, searched[i[0]])  # 1st subwalk
                    B = pos_visit(i[0], i[1], j, searched[i[0]])  # 2nd subwalk
                    C = pos_visit(i[1], habitat, step-j-k, searched[i[0]])  # 3rd subwalk
                    if A == False or B == False or C == False:  # no valid walk
                        pass
                    else:
                        for w in itt.product(A, B, C):
                            apart, bpart, cpart = w[0], w[1], w[2]
                            walk = apart + bpart + cpart
                            util = TCDutil(walk, crop_amount, tree_amount, trespass_cost,
                                           mg_tree, mg_crop, mg_water, alpha, beta, moving_cost)
                            if util >= 0:
                                if util > maxUtil:  # better than the current optimal walk
                                    maxset = {(*walk, habitat)}
                                    maxUtil = util
                                elif util == maxUtil:  # the same as the current optimal walk
                                    maxset.add((*walk, habitat))
        searched[i[0]].add(i[1])
    return maxset


'''
The followings are the two Random Walk models.
'''


def RW_1(habitat, step):
    '''
    This is Random Walk 1.

    habitat -- habitat of the agent
    step -- max steps allowed

    Return the simulated walk.
    '''
    step_left = step
    c_loc = habitat
    locls = [habitat]
    while step_left > 0:
        c_loc = random.choice(tuple(STN(c_loc, 1)))
        locls.append(c_loc)
        step_left -= 1
    return locls


def RW_2(habitat, step):
    '''
    This is Random Walk 2.

    habitat -- habitat of the agent
    step -- max steps allowed

    Return the simulated walk.
    '''
    step_left = step
    c_loc = habitat
    locls = [habitat]
    while step_left > 0:
        c_loc = random.choice(tuple(STN(c_loc, 1).intersection(STN(habitat, step_left-1))))
        locls.append(c_loc)
        step_left -= 1
    return locls


'''
The followings are our implementation of the two Myoptic Walk models
'''


def consider_util(loc, f, w, alpha, mgc, mgt, mgw):
    '''
    Myopic agent considering the utilty from foraging in a certain block

    loc -- current location
    f -- the amount of food already consumed
    w -- the amount of water already consumed
    alpha, mgc, mgt, mgw -- parameters

    return the utility of foraging in loc.
    '''
    util = f**alpha*w**(1-alpha)
    if loc in cropset:
        if crop_amount[loc] >= mgc:
            util = (f + mgc)**alpha*w**(1-alpha)
        else:
            util = (f + crop_amount[loc])**alpha*w**(1-alpha)
    elif loc in treeset:
        if tree_amount[loc] >= mgt:
            util = (f + mgt)**alpha*w**(1-alpha)
        else:
            util = (f + tree_amount[loc])**alpha*w**(1-alpha)
    elif loc in waterset:
        util = f**alpha*(w + mgw)**(1-alpha)
    return util


def forage(cropset, treeset, loc, mgc, mgt):
    '''
    Consume resources in a location and alter cropset/treeset.

    Special Note: This method changes the resources in the map, but MW1 and
    MW1 will recover the resources after producing the optimal myopic walk.
    Thus you should not call this function indepdently. The true foraging
    process happends after the search for optimal walk finishes.

    loc -- location to forage
    mgc, mgt -- marginal gain from crop and tree (same parameters as in CRM).
    '''
    if loc in cropset:
        if crop_amount[loc] >= mgc:
            crop_amount[loc] = crop_amount[loc] - mgc
        else:
            crop_amount[loc] = 0
    elif loc in treeset:
        if tree_amount[loc] >= mgt:
            tree_amount[loc] = tree_amount[loc] - mgt
        else:
            tree_amount[loc] = 0


def MW_1(cropset, treeset, hab, step, alpha, mgc, mgt, mgw):
    '''
    This is the original Myopic Walk algorithm (MW1)

    crop_amount -- a dictionary that keeps track of the amount of food in each crop block
    tree_amount -- a dictionary that keeps track of the amount of food in each tree block
    habitat -- habitat of the agent
    step -- max steps allowed
    alpha -- exponent in the utility function
    mgc, mgt, mgw -- marginal gain from crop, tree, waterset

    return the simulated walk
    '''
    step_left = step
    locls = [hab]
    c_loc = hab
    f_amount = 0
    w_amount = 0
    c_set = deepcopy(cropset)
    t_set = deepcopy(treeset)
    while step_left > 0:
        if random.randint(0, 1) == 0:  # 50% chance
            if f_amount == w_amount == 0:
                choices = STN(c_loc, 1)-{c_loc}
                c_loc = random.choice(tuple(choices))
            else:
                max_util = f_amount**alpha*w_amount**(1-alpha)  # current utility
                max_set = set()
                choices = STN(c_loc, 1).intersection(STN(hab, step_left - 1)) - {c_loc}
                if choices == set():  # returning back to habitat before the last period
                    choices = {hab}
                for i in choices:
                    temp = consider_util(i, f_amount, w_amount, alpha, mgc, mgt, mgw)
                    if temp > max_util:
                        max_util = temp
                        max_set = {i}
                    if temp == max_util:
                        max_set.add(i)
                c_loc = random.choice(tuple(max_set))
        forage(cropset, treeset, c_loc, mgc, mgt)
        locls.append(c_loc)
        step_left -= 1
    cropset = c_set  # recover cropset
    treeset = t_set  # recover treeset
    return locls


def MW_2(cropset, treeset, hab, step, alpha, mgc, mgt, mgw):
    '''
    This is the revised Myopic Walk algorithm (MW2)

    crop_amount -- a dictionary that keeps track of the amount of food in each crop block
    tree_amount -- a dictionary that keeps track of the amount of food in each tree block
    habitat -- habitat of the agent
    step -- max steps allowed
    alpha -- exponent in the utility function
    mgc, mgt, mgw -- marginal gain from crop, tree, waterset

    return the simulated walk
    '''
    step_left = step  # reduce through iteration
    locls = [hab]  # starting at the habitat
    c_loc = hab  # current location
    f_amount = 0
    w_amount = 0
    c_set = deepcopy(cropset)
    t_set = deepcopy(treeset)
    while step_left > 0:  # within time constraint
        if c_loc not in STN(c_loc, 1).intersection(STN(hab, step_left - 1)):
            if f_amount == w_amount == 0:
                choices = STN(c_loc, 1).intersection(STN(hab, step_left - 1))
                c_loc = random.choice(tuple(choices))
            else:
                max_util = f_amount**alpha*w_amount**(1-alpha)  # current utility
                max_set = set()
                choices = STN(c_loc, 1).intersection(STN(hab, step_left - 1))
                for i in choices:
                    temp = consider_util(i, f_amount, w_amount, alpha, mgc, mgt, mgw)
                    if temp > max_util:
                        max_util = temp
                        max_set = {i}
                    if temp == max_util:
                        max_set.add(i)
                c_loc = random.choice(tuple(max_set))
        elif random.randint(0, 1) == 0:  # 50% chance of moving to another location
            if f_amount == w_amount == 0:  # moving to another location does not immediately increase utility
                choices = STN(c_loc, 1).intersection(STN(hab, step_left - 1)) - {c_loc}
                if choices == set():
                    choices = {hab}
                c_loc = random.choice(tuple(choices))  # randomly move outside
            else:
                max_util = f_amount**alpha*w_amount**(1-alpha)  # current utility
                max_set = set()
                choices = STN(c_loc, 1).intersection(STN(hab, step_left - 1)) - {c_loc}
                if choices == set():  # returning back to habitat before the last period
                    choices = {hab}
                for i in choices:
                    temp = consider_util(i, f_amount, w_amount, alpha, mgc, mgt, mgw)
                    if temp > max_util:
                        max_util = temp
                        max_set = {i}
                    if temp == max_util:
                        max_set.add(i)
                c_loc = random.choice(tuple(max_set))
        forage(cropset, treeset, c_loc, mgc, mgt)
        locls.append(c_loc)
        step_left -= 1
    cropset = c_set
    treeset = t_set
    return locls
