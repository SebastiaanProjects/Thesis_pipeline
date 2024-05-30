from MAE_utils import TimeSeriesMAEEncoder, TimeSeriesMAEDecoder
from CenterNet_utils import TimeSeriesCenterNet, generate_heatmaps, generate_offset_map, generate_size_maps, manual_loss_v2, l1_loss, extract_peaks_per_class, visualize_heatmap, visualize_size_map, visualize_offset_map, evaluate_adaptive_peak_extraction
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
    print("length of train_loader",len(train_loader))
    print("length of validation_loader", len(validation_loader))
    for epoch in range(5):         
        
        model.train()
        for features, labels , durations, keypoints, labels_list in train_loader:
            optimizer.zero_grad()
            heatmap_target = generate_heatmaps(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4)
            sizemap_target = generate_size_maps(sequence_length=len(labels[0]) ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)
            offset_target = generate_offset_map(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
            mask = torch.ones(features.size(), dtype=torch.bool)
            output = model(features,mask,downsample_factor) #adding mask to match the encoder architecture
            heatmap_prediction, size_prediction, offset_prediction = output 
            size_prediction = size_prediction.squeeze(1)
            offset_prediction = offset_prediction.squeeze(1)

            #########
            #w_focal_loss = WeightedFocalLoss()#focal_loss_weight_tensor(labelled['b.int']))
            #heatmap_loss = w_focal_loss(heatmap_prediction, heatmap_target)
            heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target) #manual instantation of focalloss
            ############


            l1_loss_size = l1_loss(size_prediction, sizemap_target)
            l1_loss_offset = l1_loss(offset_prediction, offset_target) 
            total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset


            #total_loss = heatmaploss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset
            total_loss.backward()
            optimizer.step()

            writer.add_scalar('Training/size_loss', l1_loss_size.item(), epoch * len(train_loader) + fold)
            writer.add_scalar('Training/heatmap_loss', heatmap_loss.item(), epoch * len(train_loader) + fold)
            writer.add_scalar('Training/offset_loss', l1_loss_offset.item(), epoch * len(train_loader) + fold)          
            writer.add_scalar('Training/total_loss', total_loss.item(), epoch * len(train_loader) + fold) #item()

        # Validate
        model.eval()
        with torch.no_grad():
            validation_loss, heatmaploss_tot, l1_loss_offset_tot, l1_loss_size_tot = 0,0,0,0

            for features, labels , durations, keypoints, labels_list in validation_loader:
                heatmap_target = generate_heatmaps(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4)
                sizemap_target = generate_size_maps(sequence_length=len(labels[0]) ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)
                offset_target = generate_offset_map(sequence_length=len(labels[0]), batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
                mask = torch.ones(features.size(), dtype=torch.bool)
                output = model(features,mask, downsample_factor) #adding mask to match the encoder architecture
                
                heatmap_prediction, size_prediction, offset_prediction = output
                size_prediction = size_prediction.squeeze(1)
                offset_prediction = offset_prediction.squeeze(1) 
                ###
                #w_focal_loss = WeightedFocalLoss()
                #heatmaploss = w_focal_loss(heatmap_prediction, heatmap_target)
                heatmaploss = manual_loss_v2(heatmap_prediction, heatmap_target)
                ##
                #peaks = extract_peaks_per_class(heatmap_prediction, 5) # in format [batch_nr[keypoints, sizes, offsets]].
                #print(peaks)
                #peaks = process_peaks_per_class_new(heatmap_prediction, heatmap_target, window_size=20/downsample_factor)
                peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (40/downsample_factor), prominence_factor=0.75)
                print(peaks)
                print("batch item 0 peaks =", len(peaks[0]))
                print("batch item 1 peaks =", len(peaks[1]))
                print("batch item 2 peaks =", len(peaks[2]))
                print("batch item 3 peaks =", len(peaks[3]))
                print("batch item 4 peaks =", len(peaks[4]))
                print("batch item 5 peaks =", len(peaks[5]))
                

                l1_loss_size = l1_loss(size_prediction, sizemap_target)
                l1_loss_offset = l1_loss(offset_prediction, offset_target)            
                total_loss = heatmaploss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

                heatmaploss_tot+= heatmaploss
                validation_loss+= total_loss
                l1_loss_size_tot += l1_loss_size
                l1_loss_offset_tot+= l1_loss_offset

                writer.add_scalar('Validation/size_loss', l1_loss_size, epoch * len(validation_loader) + fold)
                writer.add_scalar('Validation/heatmap_loss', heatmaploss, epoch * len(validation_loader) + fold)
                writer.add_scalar('Validation/offset_loss', l1_loss_offset, epoch * len(train_loader) + fold)          
                writer.add_scalar('Validation/total_loss', total_loss, epoch * len(validation_loader) + fold)

        print(f'Fold {fold}, Epoch {epoch}, Size Loss: {l1_loss_size_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Heatmap Loss: {heatmaploss_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, offset Loss: {l1_loss_offset_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Total Validation Loss: {validation_loss / len(validation_loader)}')
    torch.set_printoptions(profile="full")  
    #visualize_size_map(sizemap_target, size_prediction[len(size_prediction)-1])
    visualize_heatmap(heatmap_target, heatmap_prediction)
    visualize_size_map(sizemap_target, size_prediction)
    visualize_offset_map(offset_target, offset_prediction)
    torch.set_printoptions(profile="default") # reset

writer.close()


#need to get a working peak_extractor. 
# I am not convinced about the current working of the peak extraction
# also I want to tryout with the weighted tensor for focal_loss (manual_loss_v2)
# When I have the peak extraction working I can work on the real results like confusion matrices 
# recall and precision and what not
# the conversion from to original label then needs to work to but that is easy
# the test can also be done then

