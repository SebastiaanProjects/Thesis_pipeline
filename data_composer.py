from Data_extraction_utils import prepare_targets ,custom_collate_fn, partition_and_mask, TimeSeriesDataset, ensure_label_representation, prepare_behaviour_data_duration, most_logical_fold, testset_occurences
import pandas as pd
import numpy as np
import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loading the unlabelled data for preatraining the MAE sequence

folder_path = r'/home/sebastiman/ProjectenFolder/big-tuna/crab plover data new' #Dear reader, fill in own pathway
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
pre_train_tensors_list = []

#everywhere there are missing speed values also in the labelless data. This means that we have to drop some rows
#fortunately when looking at the data it became appearant that the missing speed labels are consistently missing throughout the entire measurement, not interfering with other 
#measurements


for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df = df.iloc[:,2:6] #drop columns responsilbe for index, time, and speed; leaves accelarations for x y z and speed
    clusters = [df.iloc[i:i+200,:] for i in range(0, df.shape[0], 200)] #chops the dataframe into 10 second measurments (20Hz * 10 seconds = 200 datapoints)
    for cluster in clusters:
        speed_column = cluster.iloc[:,3] #check if the speed values are present otherwise remove this measurement
        #print(speed_column)
        if speed_column.isnull().any() == False:
            pre_train_tensors_list.append(torch.tensor(cluster.values, dtype=torch.float64))#, device=device))

#random.shuffle(pre_train_tensors_list) #exluding biases by hussling the orders of the items. No bias per dataset
#for pretraining, use this variable:
pre_train_tensors_list




#loading in the labelled data for the training
labelled = pd.read_csv('/home/sebastiman/ProjectenFolder/big-tuna/calibrated_and_labelled_crabplovers_pt_2.csv') #fill in own pathway
#extracting only the b.int and x y z accelerations and speed in this case. For 16 classes


train_values_list = []
train_labels_list = []
labelled = labelled[['x', 'y', 'z','speed', 'b.int']]
clusters = [labelled.iloc[i:i+200,:] for i in range(0, labelled.shape[0], 200)] #chops the dataframe into 10 second measurments (20Hz * 10 seconds = 200 datapoints)
for cluster in clusters:
    train_values_list.append(cluster[['x', 'y', 'z', 'speed']].values.tolist())
    train_labels_list.append(cluster['b.int'].values.astype(int).tolist()) 


train_datapoints, train_labels, test_datapoints, test_labels = ensure_label_representation(train_values_list, train_labels_list,max_test_size=0.185) #Stratification, max_test_size = 0.185 so trainingset = 500 (for clear folds and batchsizes) and the testset will contain all classes

labels_for_refrence = labelled['b.int']




original_features_train, original_labelling_train, durations_all_train, starts_and_endings_all_train, keypoints_all_train, labels_list_train = prepare_behaviour_data_duration(train_datapoints, train_labels, downsample_factor=1) #labels contains all the labels as structured in the original data, labels_all structures the unique labels after eachother
original_features_test, original_labelling_test, durations_all_test, starts_and_endings_all_test, keypoints_all_test, labels_list_test = prepare_behaviour_data_duration(test_datapoints, test_labels, downsample_factor=1) #labels contains all the labels as structured in the original data, labels_all structures the unique labels after eachother
sequence_length = len(original_features_train[0])

train_labels_indexed = prepare_targets(original_labelling_train, labels_for_refrence)
test_labels_indexed = prepare_targets(original_labelling_test, labels_for_refrence)
indexed_labels_list_train = prepare_targets(labels_list_train, labels_for_refrence)
indexed_labels_list_test = prepare_targets(labels_list_test, labels_for_refrence)


#finding the segements with the most behaviours in them, to help determine what a good window_size is for peak extraction
#find the behaviour which occurs most often, if the most occuring action within one segment occurs 5 times, we know that
#we can split the sequence length into roughly 5 segments to also capture the peaks for one behaviour

max_occurrence_per_segment = [max(behaviourslist.count(x) for x in set(behaviourslist)) for behaviourslist in labels_list_train]
window_size = int(sequence_length/(max(max_occurrence_per_segment)))


#trainingset is now 500 sequences. 
#by division the most logical kfold will be 10 folds of size 500.
#this leaves 113 for the testset, allowing for stratification to occur in a strict fashion

train_data_size = len(train_labels_indexed)
num_folds = int(most_logical_fold(train_data_size))
fold_size = train_data_size / num_folds
batches_within_fold = most_logical_fold(fold_size) 
batch_size = int(fold_size / batches_within_fold)

num_classes = len(labels_for_refrence.unique())

trainingset = TimeSeriesDataset(original_features_train, train_labels_indexed, durations_all_train,  keypoints_all_train, indexed_labels_list_train)
testset = TimeSeriesDataset(original_features_test, test_labels_indexed, durations_all_test,  keypoints_all_test, indexed_labels_list_test)

#for evaluating:

testset_occurences(testset)
        