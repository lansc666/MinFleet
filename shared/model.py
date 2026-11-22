"""Population-based trip demand and minimum-fleet matching."""

from collections import defaultdict
import copy
import itertools
import math
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.bipartite.matching import hopcroft_karp_matching
from scipy import stats


class ScenarioPop:
    """Sample OD pairs and departure times from population-based demand."""

    def __init__(self, grid, time_dct, tf=60, theta=2, pop_o=1, pop_d=1):
        self.grid = grid
        self.grid_num = len(grid)
        self.time_dct = time_dct
        self.tf = tf
        self.pop = np.array(grid["value"].tolist())
        self.theta = theta
        self.pop_o = pop_o
        self.pop_d = pop_d
        self.trips = None

    def generate_popnum_probability(self):
        prob_dct = {
            (o, d): 0
            for o, d in itertools.product(
                np.arange(self.grid_num), np.arange(self.grid_num)
            )
        }
        for o in range(self.grid_num):
            for d in range(self.grid_num):
                if o == d:
                    continue
                if d not in self.time_dct[o]:
                    prob_dct[o, d] = 0
                prob_dct[o, d] = (
                    (self.pop[o] ** self.pop_o)
                    * (self.pop[d] ** self.pop_d)
                    / ((self.time_dct[o][d] + 1) ** self.theta)
                )
        sum_prob = sum(list(prob_dct.values()))
        prob_dct = {(o, d): prob_dct[o, d] / sum_prob for o, d in prob_dct}
        self.prob_dct = prob_dct

    def generate_flow(self, ord_num):
        od_lst = list(self.prob_dct.keys())
        p_lst = list(self.prob_dct.values())
        o_lst = []
        d_lst = []
        ids = np.random.choice(list(range(len(od_lst))), ord_num, p=p_lst, replace=True)
        for i in ids:
            o, d = od_lst[i]
            o_lst.append(o)
            d_lst.append(d)
        df = pd.DataFrame({"o": o_lst, "d": d_lst})
        self.trips = df

    def generate_even_time(self):
        To_lst = []
        Td_lst = []
        for i, item in self.trips.iterrows():
            To = int(np.random.uniform(low=0, high=self.tf, size=1)[0])
            o, d = item["o"], item["d"]
            dt = int(self.time_dct[o][d] / 60 + 0.5)
            Td = To + dt
            To_lst.append(To)
            Td_lst.append(Td)
        self.trips["to"] = To_lst
        self.trips["td"] = Td_lst
        return self.trips.copy()

    def generate_normal_time(self, mu, sigma):
        To_lst = []
        Td_lst = []
        for i, item in self.trips.iterrows():
            To = int(
                stats.truncnorm(
                    (0 - mu) / sigma, (self.tf - mu) / sigma, loc=mu, scale=sigma
                ).rvs(1)[0]
            )
            o, d = item["o"], item["d"]
            dt = int(self.time_dct[o][d] / 60 + 0.5)
            Td = To + dt
            To_lst.append(To)
            Td_lst.append(Td)
        self.trips["to"] = To_lst
        self.trips["td"] = Td_lst
        return self.trips.copy()


def real_double_dict(N, T1, T2, time_dct):
    space_dct = defaultdict(list)
    for o in range(N):
        for d in time_dct[o]:
            min_t = math.ceil(0 - time_dct[o][d] / 60 - T2)
            max_t = math.floor(0 - time_dct[o][d] / 60 + T1)
            space_dct[o] += [(i, d) for i in range(min_t, max_t + 1)]
    return space_dct


def get_grid_ord_chain(matches):
    """Recover trip chains and count cycles from bipartite matchings."""
    matches_ = copy.deepcopy(matches)
    chains = []
    ring = 0
    for match in matches_:
        while len(match) > 0:
            match_1 = list(match.keys())[0]
            seed = match_1.split("_")[0]
            chain_lst = [seed]
            while seed + "_l" in match:
                next_ = match[seed + "_l"].split("_")[0]
                if next_ in chain_lst:
                    ring += 1
                    break
                chain_lst.append(next_)
                seed = next_
            seed = match_1.split("_")[0]
            # Extend the chain backwards through predecessor matches.
            while seed + "_r" in match:
                before_ = match[seed + "_r"].split("_")[0]
                if before_ in chain_lst:
                    break
                chain_lst.insert(0, before_)
                seed = before_
            for ch in chain_lst:
                if ch + "_r" in match:
                    del match[ch + "_r"]
                if ch + "_l" in match:
                    del match[ch + "_l"]
            chains.append([int(i) for i in chain_lst])
    return chains, ring


class OrdGraph:
    """Build trip connections and calculate fleet size by bipartite matching."""

    def __init__(self, trip, time_dct, space_dct, max_delay):
        self.trip = trip
        self.time_dct = time_dct
        self.N = len(self.time_dct)
        self.space_dct = space_dct
        self.ord_num = len(trip)
        self.fleet_size = self.ord_num
        self.max_delay = max_delay
        self.create_network()

    def create_network(self):
        ord_ = 0
        self.G = nx.DiGraph()
        self.d_hr = defaultdict(list)
        self.o_hr = defaultdict(list)
        for i, (o, d, to, td) in self.trip.iterrows():
            self.G.add_node(ord_, lo=o, ld=d, to=to - self.max_delay, td=td, wt=0)
            self.o_hr[to - self.max_delay, o].append(ord_)
            self.d_hr[td, d].append(ord_)
            ord_ += 1
        self.create_simple_hr_edges()

    def create_simple_hr_edges(self):
        for t, o in self.o_hr:
            for td, d in self.space_dct[o]:
                for ord_o, ord_d in itertools.product(
                    self.o_hr[t, o], self.d_hr[t + td, d]
                ):
                    if ord_o != ord_d:
                        self.G.add_edge(ord_d, ord_o)

    def create_bipart(self):
        left_lst = []
        right_lst = []
        for n in self.G.nodes:
            left_lst.append(str(n) + "_l")
            right_lst.append(str(n) + "_r")
        B = nx.Graph()
        B.add_nodes_from(left_lst, bipartite=0)
        B.add_nodes_from(right_lst, bipartite=1)
        eds = []
        for i, j in self.G.edges:
            eds.append((str(i) + "_l", str(j) + "_r"))
        B.add_edges_from(eds)
        self.B = B

    def delete_node(self):
        sub_n = [n for n in self.B.nodes if self.B.degree(n) != 0]
        self.B = self.B.subgraph(sub_n)

    def get_matching(self):
        components_ = nx.connected_components(self.B)
        subs_ = [self.B.subgraph(i) for i in components_]
        self.matchings = [hopcroft_karp_matching(sub) for sub in subs_]
        self.matching_chain, ring = get_grid_ord_chain(self.matchings)
        self.fleet_size = (
            self.ord_num - sum([len(m) // 2 for m in self.matchings]) + ring
        )

    def get_individual_waiting_time(self):
        """Return signed waiting times in minutes, with zero for chain starts."""
        chains = self.matching_chain
        G = self.G
        time_dct = self.time_dct
        waiting_lst = [0 for i in range(self.ord_num)]
        for chain in chains:
            for i, ord_ in enumerate(chain):
                if i == 0:
                    continue
                o = G.nodes[ord_]  # Current trip.
                d = G.nodes[chain[i - 1]]  # Preceding trip.
                ld, td = d["ld"], d["td"]
                lo, to = o["lo"], o["to"]
                waiting_time = td + time_dct[lo][ld] // 60 - to
                # Negative values mean the vehicle arrives before departure.
                waiting_lst[ord_] = waiting_time
                self.G.nodes[ord_]["wt"] = waiting_time
        self.waiting_lst = waiting_lst
        return waiting_lst

