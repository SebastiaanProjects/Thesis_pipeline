from MAE_utils import TimeSeriesMAEEncoder, TimeSeriesMAEDecoder
from CenterNet_utils import TimeSeriesCenterNet, generate_heatmaps, generate_offset_map, generate_size_maps, manual_loss_v2, l1_loss, visualize_heatmap, visualize_size_map, visualize_offset_map, evaluate_adaptive_peak_extraction, neighbouring_peaks_sort, iou_based_peak_suppression, iou_1d, reconstruct_timelines_ascending_activation, combine_peaks_with_maps, reconstruct_timelines_gaussian_support, reconstruct_timelines_start_max_activation
from Data_extraction_utils import custom_collate_fn, TimeSeriesDataset

from data_composer import trainingset, testset, labels_for_refrence, sequence_length#, pre_train_tensors_list, 

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter 
import seaborn as sns
import numpy as np
import torch.nn.functional as F
import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassRecall
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sns.set()
writer = SummaryWriter()


#trainingset = TimeSeriesDataset(original_features_train, train_labels_indexed, durations_all_train,  keypoints_all_train, indexed_labels_list_train)
trainingset
num_folds = 17
unique_labels_len = len(np.unique(labels_for_refrence))
size_contribution, offset_contribution, downsample_factor = 0.2, 1, 4

untrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.1)

num_classes = len(labels_for_refrence.unique()) # == 15
model = TimeSeriesCenterNet(untrained_encoder, num_classes=num_classes,downsampling_factor=downsample_factor, sequence_length=sequence_length) #unspecified, out of sight, interaction and shit are removed (shake too)
optimizer = optim.Adam(list(model.parameters()), lr=0.001,weight_decay=1e-5)

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)



for fold, (train_ids, test_ids) in enumerate(kfold.split(trainingset)): #trainingset already stratified
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids) #splits into traindata
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids) #splits into testdata

    # Create DataLoaders for train and validation
    train_loader = DataLoader(trainingset, batch_size=6, sampler=train_subsampler, collate_fn=custom_collate_fn)
    validation_loader = DataLoader(trainingset, batch_size=6, sampler=test_subsampler, collate_fn=custom_collate_fn)
    for epoch in range(5):         
    
        model.train()
        total_predictions = []
        total_predictions_gs = [] #for comparison with reconstruct_timeline gaussian_support
        total_predictions_sma = [] # for comparison with reconstruct_timeline_maximum_activation
        total_labels = []
        for fold, (features, labels, durations, keypoints, labels_list) in enumerate(train_loader):
            optimizer.zero_grad()
            heatmap_target = generate_heatmaps(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4)
            sizemap_target = generate_size_maps(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)
            offset_target = generate_offset_map(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
            mask = torch.ones(features.size(), dtype=torch.bool, device=device)
            output = model(features,mask,downsample_factor) #adding mask to match the encoder architecture
            heatmap_prediction, size_prediction, offset_prediction = output 
            size_prediction = size_prediction.squeeze(1)
            offset_prediction = offset_prediction.squeeze(1)

            heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target) #manual instantation of focalloss

            l1_loss_size = l1_loss(size_prediction, sizemap_target)
            l1_loss_offset = l1_loss(offset_prediction, offset_target) 
            total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

            total_loss.backward()
            #if epoch == 4:
            peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size=(40/downsample_factor), prominence_factor=0.75)               
            combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
            filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)
            reconstructed_timelines = reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length)
            total_predictions.append(reconstructed_timelines)
            total_labels.append(labels)

            total_predictions_tensor = torch.stack(total_predictions, dim=0).flatten()
            total_labels_tensor = torch.stack(total_labels, dim=0).squeeze(0).flatten()           
            accuracy_per_class = MulticlassAccuracy(average=None, num_classes=num_classes)
            average_accuracy = MulticlassAccuracy()
            recall_per_class = MulticlassRecall(average=None, num_classes=num_classes)
            average_recall = MulticlassRecall()
            accuracy_per_class.update(total_predictions_tensor, total_labels_tensor)
            average_accuracy.update(total_predictions_tensor, total_labels_tensor)
            recall_per_class.update(total_predictions_tensor, total_labels_tensor)
            average_recall.update(total_predictions_tensor, total_labels_tensor)
            #print("accuracy per class:", accuracy_per_class.compute())
            #print("recall per class:", recall_per_class.compute())

            ### Combined scalar logging for Training metrics ###
            #writer.add_scalars('Training/Metrics', {
            #    'average_precision': average_accuracy.compute(),
            #    'average_recall': average_recall.compute()
            #}, epoch * len(train_loader) + fold)
            
            ## testing which reconstruction is best, not final version: 
            total_predictions_gs.append(reconstruct_timelines_gaussian_support(filtered_peaks, sequence_length))
            total_predictions_gs_tensor = torch.stack(total_predictions_gs, dim=0).flatten()
            accuracy_per_class_gs = MulticlassAccuracy(average=None, num_classes=num_classes)
            average_accuracy_gs = MulticlassAccuracy()
            recall_per_class_gs = MulticlassRecall(average=None, num_classes=num_classes)
            average_recall_gs = MulticlassRecall()

            accuracy_per_class_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            average_accuracy_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            recall_per_class_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            average_recall_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            #print("accuracy per class gaussian support:", accuracy_per_class_gs.compute())
            #print("recall per class gaussian support:", recall_per_class_gs.compute())
            
            ### Combined scalar logging for Training metrics (gaussian support) ###
            #writer.add_scalars('Training/Metrics gaussian support', {
            #    'average_precision': average_accuracy_gs.compute(),
            #    'average_recall': average_recall_gs.compute()
            #}, epoch * len(train_loader) + fold)

            total_predictions_sma.append(reconstruct_timelines_start_max_activation(filtered_peaks, sequence_length))
            total_predictions_sma_tensor = torch.stack(total_predictions_sma, dim=0).flatten()
            accuracy_per_class_sma = MulticlassAccuracy(average=None, num_classes=num_classes)
            average_accuracy_sma = MulticlassAccuracy()
            recall_per_class_sma = MulticlassRecall(average=None, num_classes=num_classes)
            average_recall_sma = MulticlassRecall()

            accuracy_per_class_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            average_accuracy_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            recall_per_class_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            average_recall_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            #print("accuracy per class start maximum activation:", accuracy_per_class_sma.compute())
            #print("recall per class start maximum activation:", recall_per_class_sma.compute())
            
            ### Combined scalar logging for Training metrics (start maximum activation) ###
            #writer.add_scalars('Training/Metrics start maximum activation', {
            #    'average_precision': average_accuracy_sma.compute(),
            #    'average_recall': average_recall_sma.compute()
            #}, epoch * len(train_loader) + fold)

            optimizer.step()

            writer.add_scalars('Training/Accuracy', {
                'last activation average accuracy' : average_accuracy.compute(),
                'start maximum activation average accuracy' : average_accuracy_sma.compute(),
                'gaussian support average accuracy' : average_accuracy_gs.compute() 
            }, epoch * len(validation_loader) + fold)

            writer.add_scalars('Training/recall', {
                'last activation average recall' : average_recall.compute(),
                'start maximum activation average recall' : average_recall_sma.compute(),
                'gaussian support average recall' : average_recall_gs.compute() 
            }, epoch * len(validation_loader) + fold)
            
            writer.add_scalar('Training/size_loss', l1_loss_size.item(), epoch * len(train_loader) + fold)
            writer.add_scalar('Training/heatmap_loss', heatmap_loss.item(), epoch * len(train_loader) + fold)
            writer.add_scalar('Training/offset_loss', l1_loss_offset.item(), epoch * len(train_loader) + fold)          
            writer.add_scalar('Training/total_loss', total_loss.item(), epoch * len(train_loader) + fold)
            
            # Validate
            model.eval()
            with torch.no_grad():
                validation_loss, heatmaploss_tot, l1_loss_offset_tot, l1_loss_size_tot = 0,0,0,0
                total_predictions_val = []
                total_predictions_val_gs = [] # for comparison with reconstruct_timeline gaussian_support
                total_predictions_val_sma = [] # for comparison with reconstruct_timeline_maximum_activation
                total_labels_val = [] #### make a tensor with all the labels and one with all the predictions then try average recall and precision
                
                for fold, (features, labels, durations, keypoints, labels_list) in enumerate(validation_loader):
                    heatmap_target = generate_heatmaps(sequence_length=len(labels[0]), batch_size=validation_loader.batch_size, keypoints_batch=keypoints, classes_batch=labels_list, num_classes=unique_labels_len, durations_batch=durations, downsample_factor=4)
                    sizemap_target = generate_size_maps(sequence_length=len(labels[0]), batch_size=validation_loader.batch_size, keypoints_batch=keypoints, durations_batch=durations, downsample_factor=4)
                    offset_target = generate_offset_map(sequence_length=len(labels[0]), batch_size=validation_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
                    mask = torch.ones(features.size(), dtype=torch.bool, device=device)
                    output = model(features, mask, downsample_factor) # adding mask to match the encoder architecture
                    
                    heatmap_prediction, size_prediction, offset_prediction = output
                    size_prediction = size_prediction.squeeze(1)
                    offset_prediction = offset_prediction.squeeze(1)
                    heatmaploss = manual_loss_v2(heatmap_prediction, heatmap_target)

                    peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size=(40/downsample_factor), prominence_factor=0.75)
                                    
                    combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
                    filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

                    total_predictions_val.append(reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length))
                    total_labels_val.append(labels)
                    total_predictions_val_tensor = torch.stack(total_predictions_val, dim=0).flatten()
                    total_labels_tensor_val = torch.stack(total_labels_val, dim=0).flatten()
                    accuracy_per_class_val = MulticlassAccuracy(average=None, num_classes=num_classes)
                    average_accuracy_val = MulticlassAccuracy()
                    recall_per_class_val = MulticlassRecall(average=None, num_classes=num_classes)
                    average_recall_val = MulticlassRecall()

                    accuracy_per_class_val.update(total_predictions_val_tensor, total_labels_tensor_val)
                    average_accuracy_val.update(total_predictions_val_tensor, total_labels_tensor_val)
                    recall_per_class_val.update(total_predictions_val_tensor, total_labels_tensor_val)
                    average_recall_val.update(total_predictions_val_tensor, total_labels_tensor_val)
                    #print("accuracy per class:", accuracy_per_class_val.compute())
                    #print("recall per class:", recall_per_class_val.compute())
                    
                    ### Combined scalar logging for Validation metrics ###
                    #writer.add_scalars('Validation/Metrics', {
                    #    'average_precision': average_accuracy_val.compute(),
                    #    'average_recall': average_recall_val.compute()
                    #}, epoch * len(validation_loader) + fold)

                    ## testing which reconstruction is best, not final version:
                    total_predictions_val_gs.append(reconstruct_timelines_gaussian_support(filtered_peaks, sequence_length))
                    total_predictions_val_gs_tensor = torch.stack(total_predictions_val_gs, dim=0).flatten()
                    accuracy_per_class_val_gs = MulticlassAccuracy(average=None, num_classes=num_classes)
                    average_accuracy_val_gs = MulticlassAccuracy()
                    recall_per_class_val_gs = MulticlassRecall(average=None, num_classes=num_classes)
                    average_recall_val_gs = MulticlassRecall()

                    accuracy_per_class_val_gs.update(total_predictions_val_gs_tensor, total_labels_tensor_val)
                    average_accuracy_val_gs.update(total_predictions_val_gs_tensor, total_labels_tensor_val)
                    recall_per_class_val_gs.update(total_predictions_val_gs_tensor, total_labels_tensor_val)
                    average_recall_val_gs.update(total_predictions_val_gs_tensor, total_labels_tensor_val)
                    #print("accuracy per class gaussian support:", accuracy_per_class_val_gs.compute())
                    #print("recall per class gaussian support:", recall_per_class_val_gs.compute())
                    
                    ### Combined scalar logging for Validation metrics (gaussian support) ###
                    #writer.add_scalars('Validation/Metrics gaussian support', {
                    #    'average_precision': average_accuracy_val_gs.compute(),
                    #    'average_recall': average_recall_val_gs.compute()
                    #}, epoch * len(validation_loader) + fold)

                    total_predictions_val_sma.append(reconstruct_timelines_start_max_activation(filtered_peaks, sequence_length))
                    total_predictions_val_sma_tensor = torch.stack(total_predictions_val_sma, dim=0).flatten()
                    accuracy_per_class_val_sma = MulticlassAccuracy(average=None, num_classes=num_classes)
                    average_accuracy_val_sma = MulticlassAccuracy()
                    recall_per_class_val_sma = MulticlassRecall(average=None, num_classes=num_classes)
                    average_recall_val_sma = MulticlassRecall()

                    accuracy_per_class_val_sma.update(total_predictions_val_sma_tensor, total_labels_tensor_val)
                    average_accuracy_val_sma.update(total_predictions_val_sma_tensor, total_labels_tensor_val)
                    recall_per_class_val_sma.update(total_predictions_val_sma_tensor, total_labels_tensor_val)
                    average_recall_val_sma.update(total_predictions_val_sma_tensor, total_labels_tensor_val)
                    #print("accuracy per class start maximum activation:", accuracy_per_class_val_sma.compute())
                    #print("recall per class start maximum activation:", recall_per_class_val_sma.compute())
                    
                    ### Combined scalar logging for Validation metrics (start maximum activation) ###
                    #writer.add_scalars('Validation/Metrics start maximum activation', {
                    #    'average_precision': average_accuracy_val_sma.compute(),
                    #    'average_recall': average_recall_val_sma.compute()
                    #}, epoch * len(validation_loader) + fold)

                    l1_loss_size = l1_loss(size_prediction, sizemap_target)
                    l1_loss_offset = l1_loss(offset_prediction, offset_target)
                    total_loss = heatmaploss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

                    def correlation_coefficient(tensor_a, tensor_b):
                        a_mean = tensor_a - tensor_a.mean(dim=1, keepdim=True)
                        b_mean = tensor_b - tensor_b.mean(dim=1, keepdim=True)
                        numerator = (a_mean * b_mean).sum(dim=1)
                        denominator = torch.sqrt((a_mean ** 2).sum(dim=1) * (b_mean ** 2).sum(dim=1))
                        return numerator / denominator

                    heatmaploss_tot += heatmaploss
                    validation_loss += total_loss
                    l1_loss_size_tot += l1_loss_size
                    l1_loss_offset_tot += l1_loss_offset

                    writer.add_scalars('Validation/Accuracy', {
                        'last activation average accuracy' : average_accuracy_val.compute(),
                        'start maximum activation average accuracy' : average_accuracy_val_sma.compute(),
                        'gaussian support average accuracy' : average_accuracy_val_gs.compute() 
                    }, epoch * len(validation_loader) + fold)

                    writer.add_scalars('Validation/recall', {
                        'last activation average recall' : average_recall_val.compute(),
                        'start maximum activation average recall' : average_recall_val_sma.compute(),
                        'gaussian support average recall' : average_recall_val_gs.compute() 
                    }, epoch * len(validation_loader) + fold)
                    
                    writer.add_scalar('Validation/size_loss', l1_loss_size.item(), epoch * len(validation_loader) + fold)
                    writer.add_scalar('Validation/heatmap_loss', heatmaploss.item(), epoch * len(validation_loader) + fold)
                    writer.add_scalar('Validation/offset_loss', l1_loss_offset.item(), epoch * len(validation_loader) + fold)
                    writer.add_scalar('Validation/total_loss', total_loss.item(), epoch * len(validation_loader) + fold)
   

        print(f'Fold {fold}, Epoch {epoch}, Size Loss: {l1_loss_size_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Heatmap Loss: {heatmaploss_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Offset Loss: {l1_loss_offset_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Total Validation Loss: {validation_loss / len(validation_loader)}')


    torch.set_printoptions(profile="full")  
    #visualize_heatmap(heatmap_target, heatmap_prediction)
    #visualize_size_map(sizemap_target, size_prediction)
    #visualize_offset_map(offset_target, offset_prediction)
    torch.set_printoptions(profile="default") # reset 
    #AFTER FOLD 14:
    #Traceback (most recent call last):
    #File "/home/sebastiman/ProjectenFolder/Thesis_pipeline/pipeline_no_pretrain.py", line 96, in <module>
    #peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (40/downsample_factor), prominence_factor=0.75)
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #File "/home/sebastiman/ProjectenFolder/Thesis_pipeline/CenterNet_utils.py", line 593, in evaluate_adaptive_peak_extraction
    #peaks = adaptive_peak_extraction(heatmap, window_size, prominence_factor)
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #File "/home/sebastiman/ProjectenFolder/Thesis_pipeline/CenterNet_utils.py", line 570, in adaptive_peak_extraction
    #extracted_peaks = neighbouring_peaks_sort(extracted_peaks)
    #                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #File "/home/sebastiman/ProjectenFolder/Thesis_pipeline/CenterNet_utils.py", line 671, in neighbouring_peaks_sort
    #if position_list_1[0][0] == position_list_2[0][0]: # If position_list_2's class is the same as position_list_1's class
    #                            ~~~~~~~~~~~~~~~^^^
    #IndexError: list index out of range
writer.close()


#need to get a working peak_extractor. 
# I am not convinced about the current working of the peak extraction
# also I want to tryout with the weighted tensor for focal_loss (manual_loss_v2)
# When I have the peak extraction working I can work on the real results like confusion matrices 
# recall and precision and what not
# the conversion from to original label then needs to work to but that is easy
# the test can also be done then

