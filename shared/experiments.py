"""Per-city baseline and sensitivity experiments."""

import numpy as np
from .model import ScenarioPop, OrdGraph, real_double_dict
import pickle
import geopandas as gpd
from tqdm import tqdm, trange

country_path = None


def get_pop_province(province, city, param_lst, tp):
    """Run ten replicates for each gravity-parameter vector."""
    if "_" in city:
        return
    try:
        routes = pickle.load(open(country_path + f"routes/{city}_routes_con.pkl", "rb"))
        time_dct = pickle.load(
            open(country_path + f"routes/{city}_routes_sim.pkl", "rb")
        )
        pop = gpd.read_file(country_path + f"{province}/{city}_center.shp")
    except:
        return
    grid_num = len(pop)
    popsum = pop["value"].sum()
    fleet_size_lst = [[] for i in param_lst]
    space_dct_ = real_double_dict(grid_num, 0, 10, time_dct)
    for i, param in enumerate(tqdm(param_lst)):
        alpha_pop_opt, beta_pop_opt, gamma_opt = param
        sc = ScenarioPop(
            pop,
            routes,
            tf=120,
            theta=gamma_opt,
            pop_o=alpha_pop_opt,
            pop_d=beta_pop_opt,
        )
        sc.generate_popnum_probability()
        for seed in range(10):
            sc.generate_flow(int(popsum // 1000))
            trips = sc.generate_even_time()
            real_ord_g_ = OrdGraph(trips, time_dct, space_dct_, 0)
            real_ord_g_.create_bipart()
            real_ord_g_.delete_node()
            real_ord_g_.get_matching()
            real_ord_g_.get_individual_waiting_time()
            fleet_size_lst[i].append(real_ord_g_.fleet_size / real_ord_g_.ord_num)
            del real_ord_g_
    with open(
        country_path + f"{province}/{city}_fr_{tp}.pkl", "wb"
    ) as f:
        pickle.dump(np.array(fleet_size_lst), f)
    del space_dct_


def get_province_temporal(province, city, params):
    """Vary truncated-normal location mu from 0 to 120 minutes (sigma=20)."""
    if "_" in city:
        return
    try:
        routes = pickle.load(open(country_path + f"routes/{city}_routes_con.pkl", "rb"))
        time_dct = pickle.load(
            open(country_path + f"routes/{city}_routes_sim.pkl", "rb")
        )
        pop = gpd.read_file(country_path + f"{province}/{city}_center.shp")
    except:
        return
    grid_num = len(pop)
    popsum = pop["value"].sum()
    space_dct_ = real_double_dict(grid_num, 0, 10, time_dct)
    alpha_pop_opt, beta_pop_opt, gamma_opt = params
    sc = ScenarioPop(
        pop, routes, tf=120, theta=gamma_opt, pop_o=alpha_pop_opt, pop_d=beta_pop_opt
    )
    sc.generate_popnum_probability()
    fleet_size_lst = [[] for i in range(7)]
    for seed in trange(10):
        for i in range(7):
            sc.generate_flow(int(popsum // 1000))
            trips = sc.generate_normal_time(i * 20, 20)
            real_ord_g_ = OrdGraph(trips, time_dct, space_dct_, 0)
            real_ord_g_.create_bipart()
            real_ord_g_.delete_node()
            real_ord_g_.get_matching()
            real_ord_g_.get_individual_waiting_time()
            fleet_size_lst[i].append(real_ord_g_.fleet_size / real_ord_g_.ord_num)
        del real_ord_g_
    with open(
        country_path + f"{province}/{city}_fr_tmp_mean.pkl", "wb"
    ) as f:
        pickle.dump(np.array(fleet_size_lst), f)
    del space_dct_


def get_province_temporal_std(province, city, params):
    """Vary truncated-normal scale sigma from 5 to 35 minutes (mu=60)."""
    if "_" in city:
        return
    try:
        routes = pickle.load(open(country_path + f"routes/{city}_routes_con.pkl", "rb"))
        time_dct = pickle.load(
            open(country_path + f"routes/{city}_routes_sim.pkl", "rb")
        )
        pop = gpd.read_file(country_path + f"{province}/{city}_center.shp")
    except:
        return
    grid_num = len(pop)
    popsum = pop["value"].sum()
    space_dct_ = real_double_dict(grid_num, 0, 10, time_dct)
    alpha_pop_opt, beta_pop_opt, gamma_opt = params
    sc = ScenarioPop(
        pop, routes, tf=120, theta=gamma_opt, pop_o=alpha_pop_opt, pop_d=beta_pop_opt
    )
    sc.generate_popnum_probability()
    fleet_size_lst = [[] for i in range(7)]
    for seed in trange(10):
        for i in range(7):
            sc.generate_flow(int(popsum // 1000))
            trips = sc.generate_normal_time(60, i * 5 + 5)
            real_ord_g_ = OrdGraph(trips, time_dct, space_dct_, 0)
            real_ord_g_.create_bipart()
            real_ord_g_.delete_node()
            real_ord_g_.get_matching()
            real_ord_g_.get_individual_waiting_time()
            fleet_size_lst[i].append(real_ord_g_.fleet_size / real_ord_g_.ord_num)
        del real_ord_g_
    with open(
        country_path + f"{province}/{city}_fr_tmp.pkl", "wb"
    ) as f:
        pickle.dump(np.array(fleet_size_lst), f)
    del space_dct_
