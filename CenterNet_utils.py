import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init
import pandas as pd
import torch.nn as nn
from data_composer import labels_for_refrence
device = torch.device('cuda')


class TimeSeriesCenterNet(nn.Module):
    """
    Implements the TimeSeries version of the CenterNet architecture using an encoder and detection head to predict behavior classes, their durations, and precise starts in time-series data.

    Parameters:
    - encoder (nn.Module): The encoder module which transforms raw time-series data into a suitable feature space.
    - num_classes (int): The number of behavior classes.

    Forward Pass Input:
    - x (Tensor): The input time-series data of shape (batch_size, seq_len, feature_dim).
    - mask (Tensor): A binary mask indicating valid data points; not utilized in this implementation.

    Output:
    - heatmap (Tensor): Predictions of behavior occurrence probabilities at each timestep.
    - size (Tensor): Predictions of the duration of behaviors.
    - offset (Tensor): Predictions of the precise start offsets of behaviors.

    Example Usage:
        # Assume `x` is a batch of time-series data and `encoder` is a pre-trained encoder
        model = TimeSeriesCenterNet(encoder, num_classes=10)
        heatmap, size, offset = model(x, mask)
    """
    def __init__(self, encoder, num_classes, downsampling_factor, sequence_length):
        super().__init__()
        self.encoder = encoder
        self.detection_head = CenterNet1DHead(encoder.embed_dim, num_classes, downsampling_factor, sequence_length) #used to be just one detection head, so might differ now that there 's three classes. 

    def forward(self, x, mask, downsample_factor): #mask is not used for CenterNet but it is for the encoder
        encoded_features = self.encoder(x,mask)
        heatmap, size, offset = self.detection_head(encoded_features)
        return heatmap, size, offset
    
class CenterNet1DHead(nn.Module):
    """
    Implements a 1D version of the CenterNet head for predicting heatmaps, sizes, and offsets of behaviors in time-series data.

    Parameters:
    - input_dim (int): The number of input channels (typically the output dimension of an encoder).
    - num_classes (int): The number of behavior classes that the model should predict.

    Attributes:
    - heatmap_head (nn.Module): Convolutional layer to predict the heatmap for behavior localization.
    - size_head (nn.Module): Convolutional layer to predict the size or duration of the behavior.
    - offset_head (nn.Module): Convolutional layer to predict the offset for precise behavior localization.

    Forward Pass Input:
    - features (Tensor): The input tensor of shape (batch_size, input_dim, seq_len).

    Output:
    - heatmap (Tensor): A tensor of shape (batch_size, num_classes, seq_len), representing the confidence levels of behavior occurrences at each timestep.
    - size (Tensor): A tensor of shape (batch_size, 1, seq_len), representing the predicted duration of behaviors starting at each timestep.
    - offset (Tensor): A tensor of shape (batch_size, 1, seq_len), representing the predicted precise start offsets of behaviors.

    Example Usage:
        # Assume `features` is a batch of time-series data after passing through an encoder
        detection_head = CenterNet1DHead(input_dim=64, num_classes=10)
        heatmap, size, offset = detection_head(features)
    """
    def __init__(self, input_dim, num_classes, downsampling_factor, maximum_duration):
        super().__init__()
        self.num_classes = num_classes
        self.heatmap_head = HeatmapHead(input_dim, num_classes,downsample_factor=downsampling_factor)
        self.sizemap_head = SizeHead(input_dim, downsample_factor=downsampling_factor, maximum_duration=maximum_duration)
        self.offset_head = OffsetHead(input_dim, downsample_factor=downsampling_factor)

        #self.heatmap_head = nn.Conv1d(input_dim, num_classes, kernel_size=3, padding=1) #in image CenterNet all the 8 neighbours are compared, in a 1d series this translates to 3 neighbours
        #self.size_head = nn.Conv1d(input_dim, 1, kernel_size=3, padding=1) 
        #self.offset_head = nn.Conv1d(input_dim, 1, kernel_size=3, padding=1)

    def forward(self, features):
        features = features.permute(0,2,1) # Permute to shape (batch_size, input_dim, seq_len)
        heatmap = self.heatmap_head(features)
        size = self.sizemap_head(features)
        offset = self.offset_head(features)
        return heatmap, size, offset

class HeatmapHead(nn.Module):
    def __init__(self, input_dim, num_classes, downsample_factor):
        super(HeatmapHead, self).__init__()
        self.conv = nn.Conv1d(input_dim, num_classes, kernel_size=3, padding=1, stride=downsample_factor)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv(x))

class SizeHead(nn.Module):
    def __init__(self, input_dim, downsample_factor, maximum_duration):#, minimum_duration, maximum_duration):
        super(SizeHead, self).__init__()
        self.conv = nn.Conv1d(input_dim, 1, kernel_size=1, padding=0, stride=downsample_factor) #padding=1
        init.xavier_uniform_(self.conv.weight)  # Xavier initialization
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)
            #init.constant(self.conv.bias, 1)
        #self.relu = nn.ReLU()
        self.downsample_factor = downsample_factor
        self.maximum_duration = maximum_duration
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max = self.maximum_duration / self.downsample_factor
        min = 1
        #print("Input to SizeHead:", x)  # Debug: Print input
        out = self.conv(x)
        #print("Output after Conv1d:", out)  # Debug: Print output after convolution
        #out = self.relu(out)
        #print("Output after ReLU:", out)  # Debug: Print output after ReLU
        out = self.sigmoid(out)
        #print("output after sigmoid",out)
        
        #scaling
        #out = out* (self.miminum_duration + self.maximum_duration) + self.miminum_duration
        #print("max",max)
        #print("min", min)
        #print("scaling_factor", max - min)
        out = out * (max - min) + min
        #print("output after scaling", out)
        return out
        #return self.relu(self.conv(x)) #* 200 / downsample_factor #= times max size range

class OffsetHead(nn.Module):
    def __init__(self, input_dim, downsample_factor):
        super(OffsetHead, self).__init__()
        self.conv = nn.Conv1d(input_dim, 1, kernel_size=1, padding=0, stride=downsample_factor) #padding=1
    
    def forward(self, x):
        return self.conv(x)

def generate_heatmaps(batch_size, sequence_length, keypoints_batch, classes_batch, num_classes, durations_batch, spread_sigma=8, downsample_factor=1):
    downsampled_length = sequence_length // downsample_factor
    heatmaps_batch = np.zeros((batch_size, num_classes, downsampled_length))

    for b in range(batch_size):
        keypoints = keypoints_batch[b]
        classes = classes_batch[b]
        durations = durations_batch[b]

        for keypoint, clas, duration in zip(keypoints, classes, durations):
            keypoint = keypoint // downsample_factor
            duration = duration // downsample_factor
            object_size_adaptive_std = duration / spread_sigma #to make the gaussian kernel size adaptive instead of standard size

            if object_size_adaptive_std == 0:
                object_size_adaptive_std = 1  # Set a minimum standard deviation to avoid division by zero

            # Create a Gaussian peak at the keypoint position
            for i in range(downsampled_length):
                heatmaps_batch[b, clas, i] += np.exp(-((i - keypoint) ** 2) / (2 * object_size_adaptive_std ** 2))
    
    return torch.tensor(heatmaps_batch, dtype=torch.float32)

def generate_size_maps(batch_size, sequence_length, keypoints_batch, durations_batch, downsample_factor=1):
    """
    Generates a size map by directly assigning the duration values to the keypoint positions.

    :param batch_size: Number of sequences in the batch
    :param sequence_length: Length of the time series sequence
    :param keypoints_batch: List of lists of keypoints (center positions) for each object in each sequence
    :param durations_batch: List of lists of object sizes (durations) for each sequence
    :param downsample_factor: Factor by which the sequence length is downsampled
    :return: Size map tensor of shape (batch_size, downsampled_length)
    """
    downsampled_length = sequence_length // downsample_factor
    size_maps_batch = np.zeros((batch_size, downsampled_length))
    
    for b in range(batch_size):
        keypoints = keypoints_batch[b]
        durations = durations_batch[b]

        for keypoint, duration in zip(keypoints, durations):
            downsampled_keypoint = keypoint // downsample_factor
            size_maps_batch[b, int(downsampled_keypoint)] = duration // downsample_factor  # Directly assign the duration to the keypoint position
        
    return torch.tensor(size_maps_batch, dtype=torch.float32).squeeze(1)

def generate_offset_map(batch_size, sequence_length, keypoints_batch, downsample_factor=1):
    """
    Generates offset maps by calculating the difference between the true keypoint
    locations and their nearest downsampled grid locations for each sequence in the batch.

    :param batch_size: Number of sequences in the batch
    :param sequence_length: Length of the time series sequence
    :param keypoints_batch: List of lists of keypoints (center positions) for each object in each sequence
    :param downsample_factor: Factor by which the sequence length is downsampled
    :return: Offset map tensor of shape (batch_size, downsampled_length)
    """
    downsampled_length = sequence_length // downsample_factor
    offset_maps_batch = np.zeros((batch_size, downsampled_length))
    
    for b in range(batch_size):
        keypoints = keypoints_batch[b]
        
        for keypoint in keypoints:
            downsampled_keypoint = keypoint // downsample_factor
            offset = (keypoint % downsample_factor) / downsample_factor
            offset_maps_batch[b, int(downsampled_keypoint)] = offset
    
    return torch.tensor(offset_maps_batch, dtype=torch.float32)

def extract_peaks_per_class(heatmap, K=5):
    """
    Extract peaks from the heatmap using top-k selection and NMS, separately for each class.
    
    Args:
        heatmap (torch.Tensor): Heatmap of shape (batch_size, num_classes, length).
        K (int): Number of top scores to extract for each class.
        
    Returns:
        list: List of peaks for each class in the format (batch_index, class_index, position, score),
              sorted by the position (start time).
    """
    batch_size, num_classes, length = heatmap.shape
    peaks = []

    for b in range(batch_size):
        batch_peaks = []
        for c in range(num_classes):
            class_heatmap = heatmap[b, c, :]
            class_heatmap_nms = nms(class_heatmap.unsqueeze(0).unsqueeze(0), kernel=9).squeeze()

            topk_scores, topk_inds = torch.topk(class_heatmap_nms, K)
            for score, ind in zip(topk_scores, topk_inds):
                if score > 0:    # Apply threshold if needed
                    batch_peaks.append((b, c, ind, score))  # Store indices and scores as tensors

        # Sort peaks for the current batch by index (position)
        batch_peaks.sort(key=lambda x: x[2].item())  # Convert position to Python int for sorting
        peaks.extend(batch_peaks)

    return peaks

def nms(heat, kernel=9):
    """
    Apply Non-Maximum Suppression (NMS) on the heatmap.
    
    Args:
        heat (torch.Tensor): Heatmap of shape (batch_size, num_classes, length).
        kernel (int): Size of the NMS kernel.
        
    Returns:
        torch.Tensor: Heatmap after NMS of the same shape as input heatmap.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool1d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def class_specific_thresholding(heatmap, thresholds):
    
    """
    Apply class-specific thresholds to the heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap of shape (batch_size, num_classes, length).
        thresholds (list): List of thresholds for each class.
        
    Returns:
        torch.Tensor: Thresholded heatmap.

    # Example usage
    thresholds = [0.5] * 15  # Example thresholds for 15 classes
    thresholded_heatmap = class_specific_thresholding(heatmap, thresholds)
    thresholded_heatmap

    """
    batch_size, num_classes, length = heatmap.shape
    for c in range(num_classes):
        heatmap[:, c, :] = torch.where(heatmap[:, c, :] > thresholds[c], heatmap[:, c, :], torch.tensor(0.0).to(heatmap.device))
    return heatmap

def l1_loss(predicted, actual):
    """ Calculate the L1 loss between predictions and actual values """
    valid_mask = actual != 0  # Ignore zero targets
    #return (1/torch.sum(valid_mask)) * torch.nn.functional.l1_loss(predicted[valid_mask], actual[valid_mask], reduction='sum')
    return torch.nn.functional.l1_loss(predicted[valid_mask], actual[valid_mask], reduction='mean')

def visualize_offset_map(offset_map, predicted_offsets, batch_index=0):
    """
    Visualize the ground truth and predicted size maps.

    Parameters:
    - size_map (Tensor): Ground truth size map tensor with shape (batch_size, sequence_length).
    - predicted_sizes (Tensor): Predicted size map tensor with shape (batch_size, sequence_length).
    - batch_index (int): Index of the batch to visualize.
    """
    plt.figure(figsize=(12, 6))
    
    # Ground truth size map
    plt.subplot(1, 2, 1)
    plt.plot(offset_map[batch_index].cpu().numpy(), label='Ground Truth')
    plt.title('Ground Truth offset Map')
    plt.xlabel('Position')
    plt.ylabel('Size')
    plt.legend()

    
    # Predicted size map
    plt.subplot(1, 2, 2)
    plt.plot(predicted_offsets[batch_index].cpu().numpy(), label='Predicted', color='red')
    plt.title('Predicted offset Map')#          .detach()
    plt.xlabel('Position')
    plt.ylabel('Size')
    plt.legend()
    
    plt.show()

def visualize_size_map(size_map, predicted_sizes, batch_index=0):
    """
    Visualize the ground truth and predicted size maps.

    Parameters:
    - size_map (Tensor): Ground truth size map tensor with shape (batch_size, sequence_length).
    - predicted_sizes (Tensor): Predicted size map tensor with shape (batch_size, sequence_length).
    - batch_index (int): Index of the batch to visualize.
    """
    plt.figure(figsize=(12, 6))
    
    # Ground truth size map
    plt.subplot(1, 2, 1)
    plt.plot(size_map[batch_index].cpu().numpy(), label='Ground Truth')
    plt.title('Ground Truth Size Map')
    plt.xlabel('Position')
    plt.ylabel('Size')
    plt.legend()

    
    # Predicted size map
    plt.subplot(1, 2, 2)
    plt.plot(predicted_sizes[batch_index].cpu().numpy(), label='Predicted', color='red')
    plt.title('Predicted Size Map')#          .detach()
    plt.xlabel('Position')
    plt.ylabel('Size')
    plt.legend()
    
    plt.show()

def visualize_heatmap(heat_map, predicted_heatmap, batch_index=0):
    """
    Visualize the ground truth and predicted size maps as line plots.

    Parameters:
    - size_map (Tensor): Ground truth size map tensor with shape (batch_size, num_sequences, sequence_length).
    - predicted_sizes (Tensor): Predicted size map tensor with shape (batch_size, num_sequences, sequence_length).
    - batch_index (int): Index of the batch to visualize.
    """
    plt.figure(figsize=(14, 7))
    
    num_sequences = heat_map.size(1)
    
    # Colors for each sequence using a diverse colormap
    colors = plt.cm.tab20(np.linspace(0, 1, num_sequences))    
    
    # Ground truth size map
    plt.subplot(1, 2, 1)
    for i in range(num_sequences):
        plt.plot(heat_map[batch_index, i].detach().numpy(), label=f'Ground Truth {i+1}', color=colors[i]) #instead of .detach() it was .cpu()
    plt.title('Ground Truth heatmap')
    plt.xlabel('Position')
    plt.ylabel('Probability of Peak')
    plt.legend()
    
    # Predicted size map
    plt.subplot(1, 2, 2)
    for i in range(num_sequences):
        plt.plot(predicted_heatmap[batch_index, i].detach().numpy(), label=f'Predicted {i+1}', color=colors[i]) #instead of .detach() it was .cpu()
    plt.title('Predicted heatmap')
    plt.xlabel('Position')
    plt.ylabel('Probability of Peak')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def focal_loss_weight_tensor(list_of_behaviour):
    """
    Calculate the weights for focal loss based on the inverse square root of class frequencies,
    normalized so that the sum of weights equals 1. This helps to balance classes with different
    frequencies by adjusting the focal loss during training.

    Parameters:
    - list_of_behaviour (pd.Series): A pandas Series object containing all behavior labels from the dataset.

    Returns:
    - torch.Tensor: A tensor of weights for each class, where the weight is inversely proportional to the
      square root of the class frequency. These weights are used in the focal loss calculation.

    The function first calculates the occurrence of each label, then computes weights as the inverse of the square
    root of these counts, and normalizes them so their sum is 1. It ensures that less frequent classes get higher
    weights to focus more on them during model training.
    """
    occurences = list_of_behaviour.value_counts().to_dict()
    total_occurences = sum(occurences.values())

    weights = {label: 1/np.sqrt(count) for label, count in occurences.items()} # weights as inverse of the square root of class frequencies
    total_weight = sum(weights.values())
    weights = {label: weight / total_weight for label, weight in weights.items()} # make all the weights sum to 1

    label_to_index = {label: idx for idx, label in enumerate(sorted(occurences.keys()))} # make the mapping from label to index
    weight_tensor = torch.zeros(len(label_to_index), dtype=torch.float32)

    for label, weight in weights.items(): # making sure the correct weights are at the correct index
        index = label_to_index[label]
        weight_tensor[index] = weight

    return weight_tensor
#                                                           there will be an error here, run cell 5&6 and then this again
def manual_loss_v2(prediction_tensor, target_tensor, alpha=2, beta=4,weight_tensor=labels_for_refrence, class_weights_activated=False):
    """
    Compute the manual loss.
    
    Args:
        prediction_tensor (torch.Tensor): Predictions of size [batch_len, classes, positions].
        target_tensor (torch.Tensor): Targets of size [batch_len, classes, positions].
        alpha (float): Modifier for positive samples.
        beta (float): Modifier for negative samples.
    
    Returns:
        torch.Tensor: The computed loss.
    """
    epsilon = 1e-8  # Small value to avoid log(0)
    
    # Create a masking procedure to instatiate the which formula needs to be used
    pos_inds = target_tensor.eq(1).float() 
    neg_inds = target_tensor.lt(1).float()

    #create another masking procedure to give weights per class
    weights = focal_loss_weight_tensor(weight_tensor)
    weights = weights.unsqueeze(0).unsqueeze(2).repeat(1,1,prediction_tensor.size(2))

    # Positive loss
    pos_loss = -((1 - prediction_tensor) ** alpha) * torch.log(prediction_tensor + epsilon)
    pos_loss = pos_loss * pos_inds  
    if class_weights_activated: #weights will decrease the influence of heatmaploss in validation loss, but will perform stronger to combat imbalance
        pos_loss *= weights
    # Negative loss
    neg_loss = -((1 - target_tensor) ** beta) * (prediction_tensor ** alpha) * torch.log(1 - prediction_tensor + epsilon)
    neg_loss = neg_loss * neg_inds
    if class_weights_activated:
        neg_loss *= weights 
    
    # Combine positive and negative loss
    loss = pos_loss + neg_loss
    
    # Average over the batch and classes
    loss = loss.sum(dim=(1, 2))  # Sum over positions and classes
    loss = loss.mean()  #average over batch

    return loss    

def process_peaks_per_class_new(predicted_heatmap, real_heatmap, window_size=3):
    """
    Process peaks per class in the predicted heatmap and compare with the real heatmap.
    Args:
        predicted_heatmap (torch.Tensor): The predicted heatmap (batch_size, num_classes, width).
        real_heatmap (torch.Tensor): The real heatmap (batch_size, num_classes, width).
        window_size (int): The window size for NMS.
    Returns:
        List of tuples: Processed peaks with (batchnr, class, location, predicted_value, real_value).
    """
    batch_size, num_classes, width = predicted_heatmap.size()
    
    # Apply NMS and extract peaks
    heatmap_nms = torch.zeros_like(predicted_heatmap)
    for b in range(batch_size):
        for c in range(num_classes):
            heatmap_nms[b, c] = nms_1d(predicted_heatmap[b, c], window_size)
    
    peaks = extract_peaks_per_class(heatmap_nms)

    # Compare the extracted peaks with the real heatmap
    processed_peaks = []
    for batchnr, classnr, location, probability in peaks:
        real_value = real_heatmap[batchnr, classnr, location].item()
        processed_peaks.append((batchnr, classnr, location, probability, real_value))
    
    return processed_peaks

def nms_1d(heatmap, window_size=9):
    """
    Apply 1D Non-Maximum Suppression (NMS) to the heatmap.
    Args:
        heatmap (torch.Tensor): The heatmap (1D tensor).
        window_size (int): The size of the window to consider for NMS.
    Returns:
        torch.Tensor: The heatmap with suppressed values.
    """
    if heatmap.dim() == 1:
        heatmap = heatmap.unsqueeze(0)  # Ensure it's 2D: (1, length)

    half_window = int(window_size // 2)
    suppressed = heatmap.clone()

    for i in range(half_window, int(heatmap.size(1)) - half_window):
        window = heatmap[:, int(i) - half_window:int(i) + half_window + 1]
        max_val = window.max()
        if heatmap[:, int(i)] < max_val:
            suppressed[:, int(i)] = 0

    return suppressed
