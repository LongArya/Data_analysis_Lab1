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
"""
Разработайте программу, которая выполняет поиск частых наборов объектов в заданном наборе данных с помощью
алгоритма Apriori (или одной из его модификаций). Список результирующих наборов должен содержать как наборы,
так и значение поддержки для каждого набора. Параметрами программы являются набор, порог поддержки и способ
упорядочивания результирующего списка наборов (по убыванию значения поддержки или лексикографическое).
Проведите эксперименты на двух наборах из различных предметных областей. Наборы данных должны существенно отличаться
 друг от друга по количеству транзакций и/или типичной длине транзакции (количеству объектов). 
Например, наборы retail (сведения о покупках в супермаркете: см. html, скачать gzip-архив) и 
accidents (сведения о ДТП: см. детальное описание в формате PDF, см. html, скачать gzip-архив).
 В экспериментах варьируйте пороговое значение поддержки (например: 1%, 3%, 5%, 10%, 15%).

Выполните визуализацию результатов экспериментов в виде следующих диаграмм:
сравнение быстродействия на фиксированном наборе данных при изменяемом пороге поддержки;
количество частых наборов объектов различной длины на фиксированном наборе данных
 при изменяемом пороге поддержки.
 
Подготовьте отчет о выполнении задания и загрузите отчет в формате PDF в систему.
Отчет должен представлять собой связный и структурированный документ со следующими разделами:  
формулировка задания;
 
гиперссылка на каталог репозитория с исходными текстами, наборами данных и др. сопутствующими материалами; 
рисунки с результатами визуализации; 
пояснения, раскрывающие смысл полученных результатов.
"""


def test(support_thrd, sorted_method='sup'):
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
    encoder = TransactionEncoder()
    encoded_data = encoder.fit(dataset).transform(dataset)
    dataframe = pd.DataFrame(encoded_data, columns=encoder.columns_)
    print(dataframe)

    report = apriori(dataframe, min_support=support_thrd, use_colnames=True)

    if sorted_method == 'sup':
        support = report['support']
        support_order = np.argsort(support)
        report = report.reindex(report.index[support_order])
        print(report)
    return report


def plot_graph(x_points, y_points, savepath, xlabel, ylabel):
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
        report = report.sort_values(by=['itemsets'], key = lambda x: tuple(x))        
    return report


def get_association_rules(frequent_patterns, confidence_thrd, sorting_rule='support'):
    assert sorting_rule in ['lexic', 'support']
    assoc_rules = association_rules(frequent_patterns, metric='confidence', min_threshold=confidence_thrd)
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
        # save_diagrams(dataframe, MARKET_BASKET_RESULTS)
        association_rules_diagrams(dataframe, 0.05, MARKET_BASKET_RESULTS)


def get_road_traffic_accidents(demo=False, sorting_rule='support'):
    data_file = os.path.join(SCRIPT_DIR, "US_Accidents_Dec20_updated.csv")
    dataframe = pd.read_csv(data_file)
    dataframe = dataframe.iloc[:20000]
    used_cols = ['Severity', 'Street', 'City', 'State', 'Weather_Condition']
    dataframe = dataframe[used_cols]
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
        # save_diagrams(prepared_dataframe, ACCIDENTS_RESULTS)
        association_rules_diagrams(prepared_dataframe, 0.05, ACCIDENTS_RESULTS)



if __name__ == "__main__":
    print("MARKET DATASET")
    get_market_basket_analysis(demo=False, sorting_rule='lexic')
    print("ROAD TRAFFIC DATASET")
    get_road_traffic_accidents(demo=False, sorting_rule='lexic')