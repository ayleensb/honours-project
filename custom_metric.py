#import datasets
from aif360.datasets import StandardDataset
from aif360.datasets import CompasDataset
#import fairness metrics
from aif360.datasets import BinaryLabelDataset 
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric



from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import itertools 
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

#the new custom metric being proposed.
def new_metric(arr_pred, arr_true, arr_grp): 
    
    #Two arrays for privileged and not privileged 
    #g1 and g2- contain predictions and trues are lists of lists [[], []]
    grp_priv = [[], []]
    grp_unpriv = [[], []]

    #j is the number of unique groups in arr_grp 
    j = len(set(arr_grp)) #!an implicit parameter.
    
    #print("total number of unique groups: ", j)

    for i, label in enumerate(arr_grp):
        #for privileged class
        if label == 1.0:
            #add the corresponding prediction + gt label for that class using the index associated with that label
            grp_priv[0].append(arr_pred[i])
            grp_priv[1].append(arr_true[i])
        
        #for unprivileged class
        else:
            grp_unpriv[0].append(arr_pred[i])
            grp_unpriv[1].append(arr_true[i])
    
    #print("Privileged group: ", grp_priv)
    #print("Unprivileged group: ", grp_unpriv)
    
    priv_indiv_bi = [] #stores individual benefit value of each instance
    priv_grp_bi = 0 #tracks total benefit for group
    
    #1. for each index in a group calculate the benefit, bi
    for pred, gt in zip(grp_priv[0], grp_priv[1]):
        #the individual component from GEI to calculate benefit for each instance in a group
        # original bi calculation with original range of 0,1,2
        indiv_benefit = (int(pred) - int(gt)) + 1
        
        #2. Sum the total benefit of each group
        priv_grp_bi += indiv_benefit
        
        #3. divide by size of group 1 - result of this is for each class
        priv_av_bi = priv_grp_bi / len(grp_priv[0]) #this is the total number of instances in each group. [0] has predictions which will give that number

        #store individual benefit of each instance in a list
        priv_indiv_bi.append(indiv_benefit)

    #print(priv_grp_bi)
    # print(priv_av_bi)
    #print("all bi scores for privileged instances:\n", priv_indiv_bi)

    unpriv_indiv_bi = []
    unpriv_grp_bi = 0
    for pred, gt in zip(grp_unpriv[0], grp_unpriv[1]):
        indiv_benefit = (int(pred) - int(gt)) + 1
        unpriv_grp_bi += indiv_benefit
        unpriv_av_bi = unpriv_grp_bi / len(grp_priv[0])
        unpriv_indiv_bi.append(indiv_benefit)
        
    #print(unpriv_grp_bi)
    #print(unpriv_av_bi)
    #print("all bi scores for unprivileged instances:\n", unpriv_indiv_bi)

    #4. division result is divided by the sum of g1 and g2 - J
    result = (priv_av_bi + unpriv_av_bi) / j

    return result


#function to randomly generate arrays synthetically by using fixed distributions.
def array_generator(num_of_instances, grp_dist, true_dist, pred_dist, 
                    randomise=False):
    
    #a dictionary that stores the types of expected distributions and maps to their corresp probabilties
    distribution_mapping = {"50/50": (0.5, 0.5), #50% 0s 50% 1s
                            "80/20": (0.8, 0.2), #80% 0s 20% 1s
                            "90/10": (0.9, 0.1), #90% 0s 10% 1s
                            "70/30": (0.7, 0.3)}

    group_probability = distribution_mapping.get(grp_dist)
    gt_probability = distribution_mapping.get(true_dist)
    pred_probability = distribution_mapping.get(pred_dist)

    if randomise:
        arr_grp = np.random.choice([0,1], size=num_of_instances, p=group_probability)
        arr_true = np.random.choice([0,1], size=num_of_instances, p=gt_probability)
        arr_pred = np.random.choice([0,1], size=num_of_instances, p=pred_probability)

    else:
        group_zeroes =  int(num_of_instances * group_probability[0])
        group_ones =  num_of_instances - group_zeroes
    
        true_zeroes = int(num_of_instances * gt_probability[0])
        true_ones = num_of_instances - true_zeroes
    
        pred_zeroes = int(num_of_instances * pred_probability[0])
        pred_ones = num_of_instances - pred_zeroes

        arr_grp = np.array([0] * group_zeroes + [1] * group_ones)
        arr_true = np.array([0] * true_zeroes + [1] * true_ones)
        arr_pred = np.array([0] * pred_zeroes + [1] * pred_ones)
    
    return arr_grp, arr_true, arr_pred  


def balanced_accuracy(arr_true, arr_pred):
    y_true = arr_true
    y_pred = arr_pred
    return balanced_accuracy_score(y_true, y_pred)

def aif360_metric_object(num_of_instances, arr_grp, arr_true, arr_pred, seed=42):
    
    # synthetic feature data just to comply with AIF360 formatting to apply metric.
    np.random.seed(seed)
    features = pd.DataFrame({
        'feature1': np.random.rand(num_of_instances),
        'feature2': np.random.rand(num_of_instances),
        'race': np.random.randint(0, 2, num_of_instances)  # placeholder protected attribute
    })

    features['race'] = arr_grp #protected attribute to represent the group membership array
    
    #these will be the variables to store the generated arrays with varying distributions 
    #these changing arrays will show the changing score of each metric being applied 

    data_true = features.copy() #dataframe with true labels
    data_true['label'] = arr_true
    
    data_pred = features.copy() #dataframe with predicted labels
    data_pred['label'] = arr_pred
    
    # Create BinaryLabelDataset objects for true and predicted datasets
    dataset_true = BinaryLabelDataset(df=data_true, label_names=['label'], protected_attribute_names=['race'])
    dataset_pred = BinaryLabelDataset(df=data_pred, label_names=['label'], protected_attribute_names=['race'])
    
    privileged_groups = [{'race': 1}]  # represents the majority group
    unprivileged_groups = [{'race': 0}]  # represents the minority group
    
    metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    return metric
    
def automate_analysis(num_of_instances,  dist_types=[], randomise=False):
  
    #order_type = ["asc", "desc"] 
 
    results = []

    #number 43 and not convention number 42 because the latter gives a division by zero error
    if randomise:
        np.random.seed(43)
        
    #grp_order, true_order, pred_order, order_type, order_type, order_type removed.
    for (grp_dist, true_dist, pred_dist) in itertools.product(dist_types, dist_types, dist_types):
        # #generate array for current combination
        arr_grp, arr_true, arr_pred = array_generator(num_of_instances, 
                                                      grp_dist=grp_dist, 
                                                      true_dist=true_dist, 
                                                      pred_dist=pred_dist,
                                                      randomise=randomise)
        
        balanced_accuracy_score = balanced_accuracy(arr_true, arr_pred)
        standard_custom_metric = new_metric(arr_grp, arr_true, arr_pred)
        
        aif360_metric = aif360_metric_object(num_of_instances, arr_grp, arr_true, arr_pred, seed=42)
        gei_score = aif360_metric.generalized_entropy_index()
        statistical_parity_diff = aif360_metric.mean_difference()
        disparate_impact = aif360_metric.disparate_impact()
        eq_opp_diff =  aif360_metric.equal_opportunity_difference()
        # av_odds_diff = aif360_metric.average_odds_difference()
        theil_index = aif360_metric.theil_index()
        false_positive_rate = aif360_metric.false_positive_rate()

        
        results.append({
            "grp_dist":grp_dist,
            "true_dist": true_dist,
            "pred_dist": pred_dist,
            "BAC": balanced_accuracy_score,
            "Custom Metric": standard_custom_metric,                    
            "GEI": gei_score,
            "statistical_parity_diff":statistical_parity_diff,
            "disparate_impact": disparate_impact,
            "eq_opp_diff":eq_opp_diff, 
            # "av_odds_diff":av_odds_diff,
             "theil_index": theil_index,
            "false_positive_rate": false_positive_rate
        })

    #create pandas dataframe from the results list
    metrics_df = pd.DataFrame(results)
    return metrics_df

num_of_instances = 100
dist_types = ["50/50", "80/20", "90/10", "70/30"] 
fixed_metrics_scores_table = automate_analysis(num_of_instances, dist_types, randomise=False)
fixed_latex_table = fixed_metrics_scores_table.to_latex(index=False)

random_metrics_scores_table = automate_analysis(num_of_instances, dist_types, randomise=True)
random_latex_table = random_metrics_scores_table.to_latex(index=False)


def plot_metrics(df, filter_df, x_axis, y_axis, groupby_col, pred_annotations,
                      title='Group Distribution', xlabel=None, ylabel=None, xlim=None, ylim=None, figsize=(10,6)):
    # Create a boolean mask based on the filter_df
    mask = pd.Series(True, index=df.index)
    for column, value in filter_df.items():
        mask &= (df[column] == value)
    
    df_filtered = df[mask]

    #select the columns to plot
    df_filtered = df_filtered[[x_axis, y_axis, groupby_col, pred_annotations]]
    
    plt.figure(figsize=figsize)
    
    #Group by true_dist column and plot a line for each accordingly
    for group_value, group_data in df_filtered.groupby(groupby_col):
        #Sort the data so the line connects points in order
        group_data = group_data.sort_values(by=x_axis)

        #legend should show the different true label distribution lines for the plotted group distribution 
        legend_label = f" true_dist={group_value}"
        
        plt.plot(
            group_data[x_axis],
            group_data[y_axis],
            marker='o', linestyle='-',
            label=legend_label
        )
        #each datapoint should be labelled with the changing pred_dist value it corresponds to 
        for i, row in group_data.iterrows():
            plt.annotate(
                text=str(row[pred_annotations]),
                xy=(row[x_axis], row[y_axis]),
                xytext=(5, 5),
                textcoords="offset points",
                ha='left',
                va='bottom'
            )
    
    plt.xlabel(xlabel if xlabel else x_axis)
    plt.ylabel(ylabel if ylabel else y_axis)
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.show()