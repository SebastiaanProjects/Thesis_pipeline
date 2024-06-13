import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels, durations,  keypoints, labels_all):
        """
        Initialize the dataset with features, behaviour starts, durations, and labels.

        Args:
            features (np.array): The array of sensor data.
            behaviour_starts (np.array): Start indices or times for each behaviour.
            durations (np.array): Duration of each behaviour in seconds or frames.
            labels (np.array): The behaviour labels for each time step.
        """
        self.features = torch.tensor(features, dtype=torch.float32)#, device=device)
        self.labels = torch.tensor(labels, dtype=torch.long)#, device=device)
        
        self.durations = durations
        self.keypoints = keypoints
        self.labels_list = labels_all
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (self.features[index], 
                self.labels[index], 
                self.durations[index], 
                self.keypoints[index], 
                self.labels_list[index])

def partition_and_mask(data, segment_size=10, mask_percentage=0.9, pad_value=0):
    """
    Partition the input data into segments and apply a masking operation randomly to some of the segments. This function
    is useful for preparing data for input to models like Masked Autoencoders where some inputs are intentionally masked
    to create a prediction task for the model.

    Args:
        data (torch.Tensor): The input data tensor with shape [num_samples, feature_dim], where num_samples is the total
                             number of timesteps and feature_dim is the number of features per timestep.
        segment_size (int): The size of each segment to divide the data into. The last segment may be padded if the total
                            number of samples is not divisible by the segment size.
        mask_percentage (float): The percentage of segments to be masked (i.e., hidden from the model during training).
        pad_value (float): The value used to pad the data if the total number of samples is not perfectly divisible by
                           the segment size.

    Returns:
        tuple:
            masked_segments (torch.Tensor): The segments with some masked according to the mask_percentage. Masked
                                            segments are filled with `pad_value`.
            binary_mask (torch.Tensor): A binary mask indicating which segments are masked (0 for masked, 1 for unmasked).
            segments (torch.Tensor): The original data segments before any masking was applied.
    """
    
    # If the data is not perfectly divisible by the segment size, then we will apply zero padding 
    # this will allow for a more flexible usage of data for this partitioning function
    padding_size = (segment_size - data.size(0) % segment_size) % segment_size #if perfectly divisible no padding is used otherwise 0-s are added
    padded_data = torch.nn.functional.pad(data, (0, 0, 0, padding_size), value=pad_value)
    
    num_segments = padded_data.size(0) // segment_size          #amount of segments after zero padding
    segments = padded_data.view(num_segments, segment_size, -1) #initialize the segments
    
    # Generate mask indices
    num_to_mask = int(num_segments * mask_percentage)                           #amount of masked batches
    mask_indices = np.random.choice(num_segments, num_to_mask, replace=False)   #randomize the batches that are masked
    
    binary_mask = torch.ones_like(segments, dtype=torch.float32)#, device=device)  # Create a binary mask with the same shape
    masked_segments = segments.clone()  # Clone to avoid modifying the original data

    for index in mask_indices:
        binary_mask[index] = 0 #will make the masked values be excluded for further calculations by turing the value into 0
        masked_segments[index] = pad_value #amount of padding necessary (for current usecase not necessary)

    # Reshape is not needed because we create binary_mask with the same shape as segments
    # binary_mask = binary_mask.view(-1, segment_size, data.size(1))

    return masked_segments, binary_mask, segments

def custom_collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    durations = [item[2] for item in batch]
    keypoints = [item[3] for item in batch]
    labels_list = [item[4] for item in batch]
    
    return features, labels, durations, keypoints, labels_list

def prepare_targets(labels_target, all_labels): #all_labels = labelled['b.int']
    """
    Convert raw labels into a zero-indexed format suitable for model training by mapping each label
    to a corresponding index based on its appearance order in the sorted unique labels list.

    Parameters:
    - labels_target (torch.Tensor): The tensor containing the raw labels for each datapoint.
    - all_labels (pd.Series): A pandas Series object containing all behavior labels from the dataset to determine the unique labels.

    Returns:
    - torch.Tensor: A tensor where each label in labels_target is replaced by its corresponding index,
      making it suitable for use in functions like cross-entropy loss which require class indices as target.

    This function creates a dictionary mapping each unique label to an index based on its sorted position,
    then converts each label in the input tensor to its respective index. The output tensor maintains the
    shape of the input labels_target tensor but contains indices instead of raw labels.
    """
    single_labels = np.sort(all_labels.unique()) #sort all the behaviours to later sort them without blankspaces
    dict_id = {label: index for index,label in enumerate(single_labels)} #dictionairy for behaviours to index positioning
    for idx, target_sequence in enumerate(labels_target):
        labels_target[idx] = [dict_id[int(target_item)] for target_item in target_sequence]
    
    #labels_tensor = torch.tensor(mapped_labels, dtype=torch.long).view(labels_target.size(0), labels_target.size(1))
    return labels_target

def ensure_label_representation(sequences, labels, test_size=0.1, random_state=1, max_test_size=0.17):
    """
    Ensures that each unique label is represented in the test set according to a minimum proportion while respecting an upper limit on the test set size.

    Parameters:
    - sequences (list): A list of sequences from which the train and test sets will be created.
    - labels (list of lists): A list of label lists corresponding to each sequence, where each inner list contains labels for the sequence.
    - test_size (float, optional): The minimum fraction of each label to be included in the test set, default is 0.1.
    - random_state (int, optional): Seed value for random operations to ensure reproducibility, default is 42.
    - max_test_size (float, optional): The maximum allowable fraction of the total dataset to be included in the test set, default is 0.15.

    Returns:
    - train_sequences (list): The list of sequences allocated to the training set.
    - train_labels (list of lists): The labels corresponding to the train_sequences.
    - test_sequences (list): The list of sequences allocated to the test set.
    - test_labels (list of lists): The labels corresponding to the test_sequences.

    The function maps each label to its respective sequences to ensure balanced representation. If the initial split based on `test_size` results in a test set larger than `max_test_size`, the test set is randomly reduced to meet the `max_test_size` requirement. This method ensures that each label is present in the test set but might adjust the actual test set size to not exceed the maximum specified limit.
    """
    np.random.seed(random_state)
    unique_labels = {label for seq in labels for label in seq}
    label_to_indices = {label: [] for label in unique_labels}  # Map indices to labels
    
    for index, label_seq in enumerate(labels):
        unique_in_seq = set(label_seq)
        for label in unique_in_seq:
            label_to_indices[label].append(index)

    # Calculate minimum number of samples needed in the test to have at least one of each label
    min_test_samples_per_label = {label: max(1, int(len(indices) * test_size)) for label, indices in label_to_indices.items()}
    
    # Initial test set selection to ensure representation
    test_indices = set()
    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        test_indices.update(indices[:min_test_samples_per_label[label]])

    # Adjust test set size if necessary
    if len(test_indices) / len(sequences) > max_test_size:
        test_indices = set(np.random.choice(list(test_indices), size=int(len(sequences) * max_test_size), replace=False))

    all_indices = set(range(len(sequences)))
    train_indices = list(all_indices - test_indices)
    test_indices = list(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_sequences = [sequences[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    return train_sequences, train_labels, test_sequences, test_labels

def prepare_behaviour_data_duration(original_features, original_labelling, downsample_factor=1): #NOW CONVERTING EVERYTHING INTO A TENSOR 
    durations_all = []
    starts_and_endings_all = []
    labels_all = []
    keypoints_all = []
    for index, measurement in enumerate(original_labelling): 
        current_behaviour = None
        starttime = 0
        durations = []
        starts_and_endings = []
        labels = []
        keypoints = []
        endtime = 0
        for i in range(len(measurement)):
            if measurement[i] != current_behaviour:                 
                if current_behaviour != None:
                    endtime = (i - 1)
                    duration = (endtime - starttime) / downsample_factor
                    durations.append(duration)

                    start_and_end = ((starttime / downsample_factor), (endtime / downsample_factor))
                    starts_and_endings.append(start_and_end)

                    keypoint = starttime + duration / 2 / downsample_factor      #center
                    keypoints.append(keypoint)

                    labels.append(current_behaviour)
                current_behaviour = measurement[i]
                starttime = i
        #duration is now a list for all the durations in labels[index]. To get them all together make one big list that encapsulates them all. 
        if current_behaviour is not None:
            duration = (len(measurement) - starttime) / downsample_factor
            durations.append(duration)

            start_and_end = ((starttime / downsample_factor), (len(measurement) / downsample_factor))
            starts_and_endings.append(start_and_end) 

            keypoint = starttime + ((len(measurement) - starttime)/ 2) / downsample_factor      #center
            keypoints.append(keypoint) 

            labels.append(current_behaviour)     

        durations_all.append(durations)
        starts_and_endings_all.append(starts_and_endings)
        keypoints_all.append(keypoints)
        labels_all.append(labels)

    #label transfer function to index

    return original_features, original_labelling, durations_all, starts_and_endings_all, keypoints_all, labels_all #labels contains all the labels as structured in the original data, labels_all structures the unique labels after eachother

def most_logical_fold(trainingsetsize):
    #usually the number of folds. between 5 or 10 is most preferable, see what's closest to that interval
    divide_options = []
    for i in np.arange(1, trainingsetsize):
        if trainingsetsize%i ==0:
            if i <= 15 and i > 5:
                return i
            
def testset_occurences(testset):
    """I have created the evaluation metrics to work on macro precision, 
        to calculate the micro precisions the occurences of each class are necessary.
        Therefor this is the counter for when testset is added"""
    all_labels = [testset[i][1].tolist() for i in range(len(testset))]
    all_labels = [item for sublist in all_labels for item in sublist]
    label_frame = pd.DataFrame(all_labels, columns=['labels'])
    print("Occurrences of each class:")
    print(label_frame['labels'].value_counts())
    return label_frame['labels'].value_counts().to_dict()