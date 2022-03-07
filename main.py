import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15]
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MARKET_BASKET_RESULTS = os.path.join(RESULTS_DIR, "market_basket")
ACCIDENTS_RESULTS = os.path.join(RESULTS_DIR, "accidents")
ASSOC_RULES_THRESHOLDS = np.concatenate((np.array([0.2, 0.3, 0.35, 0.4]), np.arange(0.75, 1, 0.05)))

os.makedirs(MARKET_BASKET_RESULTS, exist_ok=True)
os.makedirs(ACCIDENTS_RESULTS, exist_ok=True)


def plot_graph(x_points, y_points, savepath, xlabel, ylabel):
    # plots graph
    fig = plt.figure(figsize=(15, 8))
    plt.plot(x_points, y_points, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savepath)
    plt.close()


def get_frequent_patterns_from_dataframe(dataframe, support_thrd, sorting_rule="support"):
    assert sorting_rule in ["lexic", "support"]
    report = apriori(dataframe, min_support=support_thrd, use_colnames=True)
    if sorting_rule == 'support':
        report = report.sort_values(by=['support'], ascending=False)
    else:
        # itemsets are stored as the stored sets, so cast to tuple is required for correct sorting
        report = report.sort_values(by=['itemsets'], key = lambda x: tuple(x))        
    return report


def get_association_rules(frequent_patterns, confidence_thrd, sorting_rule='support'):
    assert sorting_rule in ['lexic', 'support']
    assoc_rules = association_rules(frequent_patterns, metric='confidence', min_threshold=confidence_thrd)
    # keep only columns that are required in task description
    assoc_rules = assoc_rules[['antecedents', 'consequents', 'support', 'confidence']]
    if sorting_rule == 'support':
        assoc_rules = assoc_rules.sort_values(by=['support'], ascending=False)
    else:
        assoc_rules = assoc_rules.sort_values(by=['antecedents', 'consequents'], key = lambda x: tuple(x))
    return assoc_rules


def save_diagrams(dataframe, root_dir):
    found_sets_nums = []
    spent_time = []
    for thrd in THRESHOLDS:
        start_time = time.time()
        report = get_frequent_patterns_from_dataframe(dataframe, thrd)
        finish_time = time.time()
        found_sets_nums.append(report.shape[0])
        spent_time.append(finish_time - start_time)
    found_sets_scheme_path = os.path.join(root_dir, "frequent_sets_nums.png")
    time_scheme_path = os.path.join(root_dir, "spent_time.png")
    plot_graph(THRESHOLDS, found_sets_nums, found_sets_scheme_path,
               "min support threshold", "found frequent sets")
    plot_graph(THRESHOLDS, spent_time, time_scheme_path,
               "min support threshold", "spent time (s)")


def association_rules_diagrams(dataframe, min_support_thrd, root_dir):
    found_rules_nums = []
    spent_time_stat = []
    frequent_itemsets = get_frequent_patterns_from_dataframe(dataframe, min_support_thrd)
    for conf_thrd in tqdm(ASSOC_RULES_THRESHOLDS):
        start_time = time.time()
        assoc_rules = get_association_rules(frequent_itemsets, conf_thrd)
        finish_time = time.time()
        found_rules_nums.append(assoc_rules.shape[0])        
        spent_time = finish_time - start_time
        spent_time_stat.append(spent_time)
    time_scheme_path = os.path.join(root_dir, "assoc_rules_time_dependency.png")
    found_rules_sheme_path = os.path.join(root_dir, "found_rules_thrd_dependency.png")
    plot_graph(ASSOC_RULES_THRESHOLDS, found_rules_nums, found_rules_sheme_path,
               "confidence threshold", f"found rules, support = {min_support_thrd}")
    plot_graph(ASSOC_RULES_THRESHOLDS, spent_time_stat, time_scheme_path, 
               "confidence threshold", "spent_time")        


def get_market_basket_analysis(demo=False, sorting_rule='support'):
    data_file = os.path.join(SCRIPT_DIR, "Market_Basket_Optimisation.csv")
    with open(data_file, "r") as f:
        lines = f.readlines()
    basket_data = [line.strip('\n').split(',') for line in lines]

    encoder = TransactionEncoder()
    encoded_data = encoder.fit(basket_data).transform(basket_data)
    dataframe = pd.DataFrame(encoded_data, columns=encoder.columns_)
    if demo:
        frequent_itemsets = get_frequent_patterns_from_dataframe(dataframe, 0.05, sorting_rule=sorting_rule)
        print(f"Frequent itemsets:\n {frequent_itemsets}")
        rules = get_association_rules(frequent_itemsets, 0.1, sorting_rule=sorting_rule)        
        print(f"Association rules:\n {rules}")
    else:
        save_diagrams(dataframe, MARKET_BASKET_RESULTS)
        association_rules_diagrams(dataframe, 0.05, MARKET_BASKET_RESULTS)


def get_road_traffic_accidents(demo=False, sorting_rule='support'):
    data_file = os.path.join(SCRIPT_DIR, "US_Accidents_Dec20_updated.csv")
    dataframe = pd.read_csv(data_file)
    dataframe = dataframe.iloc[:20000]
    used_cols = ['Severity', 'Street', 'City', 'State', 'Weather_Condition']
    dataframe = dataframe[used_cols]

    #  Mlxtend library requires data to be presented in the list of lists form 
    list_of_values = dataframe.values.tolist()
    for i in range(len(list_of_values)):
        list_of_values[i][0] = 'Severity ' + str(list_of_values[i][0])
        filtered_list = []
        for j in range(len(list_of_values[i])):
            if list_of_values[i][j] is not np.NaN:
                filtered_list.append(list_of_values[i][j])
        list_of_values[i] = filtered_list

    encoder = TransactionEncoder()
    encoded_data = encoder.fit(list_of_values).transform(list_of_values)
    prepared_dataframe = pd.DataFrame(encoded_data, columns=encoder.columns_)
    if demo:
        frequent_itemsets = get_frequent_patterns_from_dataframe(prepared_dataframe, 0.05, sorting_rule=sorting_rule)
        print(f"Frequent itemsets:\n {frequent_itemsets}")
        rules = get_association_rules(frequent_itemsets, 0.1, sorting_rule=sorting_rule)        
        print(f"Association rules:\n {rules}")
    else:
        save_diagrams(prepared_dataframe, ACCIDENTS_RESULTS)
        association_rules_diagrams(prepared_dataframe, 0.05, ACCIDENTS_RESULTS)


if __name__ == "__main__":
    """
    Both functions work in 2 modes: 
    if demo is True then results are printed 
    else diagrams are plotted and saved 
    """
    print("MARKET DATASET")
    get_market_basket_analysis(demo=True, sorting_rule='support')
    print("ROAD TRAFFIC DATASET")
    get_road_traffic_accidents(demo=True, sorting_rule='lexic')