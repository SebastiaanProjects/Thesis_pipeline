from MAE_utils import TimeSeriesMAEEncoder, TimeSeriesMAEDecoder
from CenterNet_utils import TimeSeriesCenterNet, generate_heatmaps, generate_offset_map, generate_size_maps, manual_loss_v2, l1_loss, extract_peaks_per_class, visualize_heatmap, visualize_size_map, visualize_offset_map, process_peaks_per_class_new, evaluate_adaptive_peak_extraction, combine_peaks_with_maps, iou_based_peak_suppression, reconstruct_timelines_ascending_activation
from Data_extraction_utils import custom_collate_fn, TimeSeriesDataset
from pre_train import pretrained_encoder
from data_composer import trainingset, testset, labels_for_refrence, sequence_length, num_classes, num_folds, train_data_size #, pre_train_tensors_list, 

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter 
import seaborn as sns
import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassRecall
from sklearn.metrics import precision_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sns.set()
writer = SummaryWriter()


#trainingset = TimeSeriesDataset(original_features_train, train_labels_indexed, durations_all_train,  keypoints_all_train, indexed_labels_list_train)
unique_labels_len = len(np.unique(labels_for_refrence))
size_contribution, offset_contribution, downsample_factor = 0.2, 1, 4

model = TimeSeriesCenterNet(pretrained_encoder, num_classes=num_classes,downsampling_factor=downsample_factor, sequence_length=sequence_length) #unspecified, out of sight, interaction and shit are removed (shake too)
optimizer = optim.Adam(list(model.parameters()), lr=0.001,weight_decay=1e-5)

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
batch_size = train_data_size/num_folds

for fold, (train_ids, test_ids) in enumerate(kfold.split(trainingset)): #trainingset already stratified
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids) #splits into traindata
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids) #splits into testdata

    # Create DataLoaders for train and validation
    train_loader = DataLoader(trainingset, batch_size=batch_size, sampler=train_subsampler, collate_fn=custom_collate_fn)#, pin_memory=True) #pin_memory=True doesn't work if the oringal input is already stored in device
    validation_loader = DataLoader(trainingset, batch_size=batch_size, sampler=test_subsampler, collate_fn=custom_collate_fn)#, pin_memory=True)
    print("length of train_loader",len(train_loader))
    print("length of validation_loader", len(validation_loader))
    for epoch in range(5):         
        
        model.train()
        total_predictions = []
        total_labels = []
        for features, labels , durations, keypoints, labels_list in train_loader:
            optimizer.zero_grad()
            heatmap_target = generate_heatmaps(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4)
            sizemap_target = generate_size_maps(sequence_length=len(labels[0]) ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)
            offset_target = generate_offset_map(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
            mask = torch.ones(features.size(), dtype=torch.bool)#, device=device)
            output = model(features,mask,downsample_factor) #adding mask to match the encoder architecture
            heatmap_prediction, size_prediction, offset_prediction = output 
            size_prediction = size_prediction.squeeze(1)
            offset_prediction = offset_prediction.squeeze(1)

            heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target) #manual instantation of focalloss


            l1_loss_size = l1_loss(size_prediction, sizemap_target)
            l1_loss_offset = l1_loss(offset_prediction, offset_target) 
            total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

            total_loss.backward()
            if epoch == 4:
                peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (40/downsample_factor), prominence_factor=0.75)               
                combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
                filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)
                reconstructed_timelines = reconstruct_timelines(filtered_peaks, sequence_length)
                total_predictions.append(reconstructed_timelines)
                total_labels.append(labels)          
                total_predictions_tensor = torch.stack(total_predictions, dim=0).flatten()
                total_labels_tensor = torch.stack(total_labels, dim=0).flatten()
                accuracy_per_class = MulticlassAccuracy(average=None, num_classes=num_classes)
                average_accuracy = MulticlassAccuracy()
                recall_per_class = MulticlassRecall(average=None, num_classes=num_classes)
                average_recall = MulticlassRecall()

                accuracy_per_class.update(total_predictions_tensor, total_labels_tensor)
                average_accuracy.update(total_predictions_tensor, total_labels_tensor)
                recall_per_class.update(total_predictions_tensor, total_labels_tensor)
                average_recall.update(total_predictions_tensor, total_labels_tensor)
                print("accuracy per class:", accuracy_per_class.compute())
                print("recall per class:", recall_per_class.compute())

                writer.add_scalar('Training/ average precision', average_accuracy.compute(), fold)
                writer.add_scalar('Training/ average recall', average_recall.compute(), fold)


            optimizer.step()
            
            writer.add_scalar('Training/size_loss', l1_loss_size.item(), epoch * len(train_loader) + fold)
            writer.add_scalar('Training/heatmap_loss', heatmap_loss.item(), epoch * len(train_loader) + fold)
            writer.add_scalar('Training/offset_loss', l1_loss_offset.item(), epoch * len(train_loader) + fold)          
            writer.add_scalar('Training/total_loss', total_loss.item(), epoch * len(train_loader) + fold) #item()

        # Validate
        model.eval()
        with torch.no_grad():
            validation_loss, heatmaploss_tot, l1_loss_offset_tot, l1_loss_size_tot = 0,0,0,0
            total_predictions = []
            total_labels= [] ####make a tensor with all the labels and one with all the predictions then try average recall and precision
            for features, labels , durations, keypoints, labels_list in validation_loader:
                heatmap_target = generate_heatmaps(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4)
                sizemap_target = generate_size_maps(sequence_length=len(labels[0]) ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)
                offset_target = generate_offset_map(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
                mask = torch.ones(features.size(), dtype=torch.bool)#, device=device)
                output = model(features,mask, downsample_factor) #adding mask to match the encoder architecture
                
                heatmap_prediction, size_prediction, offset_prediction = output
                size_prediction = size_prediction.squeeze(1)
                offset_prediction = offset_prediction.squeeze(1) 
                heatmaploss = manual_loss_v2(heatmap_prediction, heatmap_target)                

                peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (40/downsample_factor), prominence_factor=0.75)
                                
                combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
                filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

                # (class_nr, position, activation, size, offset)
                if epoch == 4:
                    print("batch item 0 peaks =", len(filtered_peaks[0]))
                    print("batch item 1 peaks =", len(filtered_peaks[1])) #2 is quite low if constistent, but possible
                    print("batch item 2 peaks =", len(filtered_peaks[2]))
                    print("batch item 3 peaks =", len(filtered_peaks[3]))
                    print("batch item 4 peaks =", len(filtered_peaks[4]))
                    print("batch item 5 peaks =", len(filtered_peaks[5]))
                    total_predictions.append(reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length))
                    total_labels.append(labels)
                    total_predictions_tensor = torch.stack(total_predictions, dim=0).flatten()
                    total_labels_tensor = torch.stack(total_labels, dim=0).flatten()
                    accuracy_per_class = MulticlassAccuracy(average=None, num_classes=num_classes)
                    average_accuracy = MulticlassAccuracy()
                    recall_per_class = MulticlassRecall(average=None, num_classes=num_classes)
                    average_recall = MulticlassRecall()

                    accuracy_per_class.update(total_predictions_tensor, total_labels_tensor)
                    average_accuracy.update(total_predictions_tensor, total_labels_tensor)
                    recall_per_class.update(total_predictions_tensor, total_labels_tensor)
                    average_recall.update(total_predictions_tensor, total_labels_tensor)
                    print("accuracy per class:", accuracy_per_class.compute())
                    print("recall per class:", recall_per_class.compute())
                    
                    writer.add_scalar('Validation/average precision', average_accuracy.compute(), fold)
                    writer.add_scalar('Validation/average recall', average_recall.compute(), fold)

                if fold > 10:
                    torch.set_printoptions(profile="full")
                    print("combined peaks[0]", combined_peaks[0])  
                    print("filtered peaks[0]", filtered_peaks[0])
                    print("reconstructed_timelines[0]",reconstructed_timelines[0])
                    print("labels[0]", labels[0])                   

                    torch.set_printoptions(profile="default") # reset               

                l1_loss_size = l1_loss(size_prediction, sizemap_target)
                l1_loss_offset = l1_loss(offset_prediction, offset_target)            
                total_loss = heatmaploss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

                heatmaploss_tot+= heatmaploss
                validation_loss+= total_loss
                l1_loss_size_tot += l1_loss_size
                l1_loss_offset_tot+= l1_loss_offset
                
                #writer.add_scalar('validationcosine_similarity', cosine_similarity,epoch * len(validation_loader) + fold )
                writer.add_scalar('Validation/size_loss', l1_loss_size, epoch * len(validation_loader) + fold)
                writer.add_scalar('Validation/heatmap_loss', heatmaploss, epoch * len(validation_loader) + fold)
                writer.add_scalar('Validation/offset_loss', l1_loss_offset, epoch * len(train_loader) + fold)          
                writer.add_scalar('Validation/total_loss', total_loss, epoch * len(validation_loader) + fold)

        print(f'Fold {fold}, Epoch {epoch}, Size Loss: {l1_loss_size_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Heatmap Loss: {heatmaploss_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, offset Loss: {l1_loss_offset_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Total Validation Loss: {validation_loss / len(validation_loader)}')

writer.close()


#need to get a working peak_extractor. 
# I am not convinced about the current working of the peak extraction
# also I want to tryout with the weighted tensor for focal_loss (manual_loss_v2)
# When I have the peak extraction working I can work on the real results like confusion matrices 
# recall and precision and what not
# the conversion from to original label then needs to work to but that is easy
# the test can also be done then
