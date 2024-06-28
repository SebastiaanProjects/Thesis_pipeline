from Data_extraction_utils import prepare_targets ,custom_collate_fn, partition_and_mask, TimeSeriesDataset, ensure_label_representation, prepare_behaviour_data_duration, most_logical_fold, testset_occurences
from MAE_utils import TimeSeriesMAEEncoder, TimeSeriesMAEDecoder, clip_and_threshold_gradients, ClipConstraint
from CenterNet_utils import TimeSeriesCenterNet, generate_heatmaps, generate_offset_map, generate_size_maps, manual_loss_v2, l1_loss, visualize_heatmap, visualize_size_map, visualize_offset_map, evaluate_adaptive_peak_extraction, neighbouring_peaks_sort, iou_based_peak_suppression, iou_1d, reconstruct_timelines_ascending_activation, combine_peaks_with_maps, reconstruct_timelines_gaussian_support, reconstruct_timelines_start_max_activation, plot_confusion_matrix, plot_bar_chart, focal_loss_weight_tensor, save_model, load_model, check_for_nans
#from pre_train import pretrained_encoder
import pandas as pd
import numpy as np
import torch
import glob
import os
import random
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter 
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torcheval.metrics import MulticlassAccuracy, MulticlassRecall, MulticlassPrecisionRecallCurve
from sklearn.metrics import precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import copy


sns.set()
writer = SummaryWriter()
print(f"device:{device}")


#loading in the labelled data for the supervised training
seagull = pd.read_csv('/home/sebastiman/ProjectenFolder/big-tuna/Copy of combined_s_w_m_j.csv', sep=',', header=None)
seagull.rename(columns={0:'tag', 1 : 'Date_time', 2: 'id_i_think', 3: 'b.int', 4: 'x', 5:'y', 6: 'z', 7: 'speed'}, inplace=True)
seagull = seagull[seagull['b.int']!=7] #7 is  "other behaviour" , remove for soudness
seagull.loc[seagull['b.int'] > 7, 'b.int'] = seagull['b.int'] - 1 #substract the values above 7 by one

train_values_list = []
train_labels_list = []
labelled = seagull[['x', 'y', 'z','speed', 'b.int']]
print(len(labelled))
clusters = [labelled.iloc[i:i+20,:] for i in range(0, labelled.shape[0], 20)] #chops the dataframe into 1 second measurements (20Hz * 1 seconds = 20 datapoints)
for index, cluster in enumerate(clusters):
    if cluster['b.int'].nunique()>1: #only one behaviour per measurement is used in this dataset
        print(f"multiple behaviours at {index} + {cluster['b.int'].unique()}")
    train_values_list.append(cluster[['x', 'y', 'z', 'speed']].values.tolist())
    train_labels_list.append(cluster['b.int'].values.astype(int).tolist()) 


print(f"len_train_labels_list {len(train_labels_list)}")
train_datapoints, train_labels, test_datapoints, test_labels = ensure_label_representation(train_values_list, train_labels_list,max_test_size=0.30, test_size=0.1075) #Stratification, max_test_size = 0.185 so trainingset = 500 (for clear folds and batchsizes) and the testset will contain all classes
print(f"len train_labels {len(train_labels)}")
print(f"len test labels {len(test_labels)}")
original_features_train, original_labelling_train, durations_all_train, starts_and_endings_all_train, keypoints_all_train, labels_list_train = prepare_behaviour_data_duration(train_datapoints, train_labels, downsample_factor=1) #labels contains all the labels as structured in the original data, labels_all structures the unique labels after eachother
original_features_test, original_labelling_test, durations_all_test, starts_and_endings_all_test, keypoints_all_test, labels_list_test = prepare_behaviour_data_duration(test_datapoints, test_labels, downsample_factor=1) #labels contains all the labels as structured in the original data, labels_all structures the unique labels after eachother
labels_for_refrence = labelled['b.int']
sequence_length = len(original_features_train[0]) #20
num_classes = len(labels_for_refrence.unique()) #9
print(f"focal_loss_weight_tensor{focal_loss_weight_tensor(labels_for_refrence)}")

train_labels_indexed = prepare_targets(original_labelling_train, labels_for_refrence)
test_labels_indexed = prepare_targets(original_labelling_test, labels_for_refrence)
indexed_labels_list_train = prepare_targets(labels_list_train, labels_for_refrence)
indexed_labels_list_test = prepare_targets(labels_list_test, labels_for_refrence)


max_occurrence_per_segment = np.max([max(behaviourslist.count(x) for x in set(behaviourslist)) for behaviourslist in labels_list_train]) #1
window_size = sequence_length #we established that there is only one behaviour per segment, thus the window can range the entire segment

train_data_size = len(train_labels_indexed) # ==3900
num_folds = 10
fold_size = int(train_data_size / num_folds)
batches_within_fold = 30
batch_size = int(fold_size / batches_within_fold)
print(f"batchsize{batch_size}")
trainingset = TimeSeriesDataset(original_features_train, train_labels_indexed, durations_all_train,  keypoints_all_train, indexed_labels_list_train)
testset = TimeSeriesDataset(original_features_test, test_labels_indexed, durations_all_test,  keypoints_all_test, indexed_labels_list_test)
print(f"len(testset){len(testset)}")
print(f"testset_occurences{testset_occurences(testset)}")


###pretraining; OPTIONAL; IF UNWANTED COMMENT OUT

pretrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.1, sequence_length=sequence_length)
decoder = TimeSeriesMAEDecoder(embed_dim=4, decoder_embed_dim=64, num_heads=16, num_layers=1, max_seq_length=10, dropout_rate=0.1)#.to(device) #num_mask_tokens=6. 0.1 like in literature

###
optimizer = optim.Adam(list(pretrained_encoder.parameters()) + list(decoder.parameters()), lr=0.001,weight_decay=1e-5) #MAE are scalable learners uses Adam, weight_decay is regularisation
pretrained_encoder, _ = load_model(pretrained_encoder, optimizer,epoch=20, mskpct=0.5)


folder_path = r'/home/sebastiman/ProjectenFolder/big-tuna/crab plover data new' #Dear reader, fill in own pathway
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
pre_train_tensors_list = []


for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df = df.iloc[:,2:6] #drop columns responsilbe for index, time, and speed; leaves accelarations for x y z and speed
    clusters = [df.iloc[i:i+200,:] for i in range(0, df.shape[0], 200)] #chops the dataframe into 10 second measurments (20Hz * 10 seconds = 200 datapoints)
    for cluster in clusters:
        speed_column = cluster.iloc[:,3] #check if the speed values are present otherwise remove this measurement
        if speed_column.isnull().any() == False:
            #cluster.iloc[:,3] = cluster.iloc[:,3] / 20.075
            pre_train_tensors_list.append(torch.tensor(cluster.values, dtype=torch.float64))#, device=device))

#seagull['speed'] = seagull['speed'] / 22.3 #the owner fo the data told me to normalize the data like so
chunk_over_5_loss =[46, 167, 250, 299, 301, 302, 340, 360, 392, 393, 394, 491, 492, 493, 611, 615, 695, 699, 757, 814, 821, 844, 871, 907, 908, 909, 910, 911, 912, 913, 918, 946, 988, 999, 1008, 1023, 1026, 1027, 1066, 1084, 1143, 1144, 1220, 1221, 1222, 1533, 1776, 2003, 2328, 2402, 2407, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2646, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2690, 2691, 2692, 2706, 2707, 2708, 2709, 2710, 2711, 2857, 2874, 3004, 3005, 3040, 3049, 3080, 3091, 3098, 3253, 3254, 3255, 3256, 3257, 3258, 3266, 3286, 3291, 3319, 3369, 3379, 3422, 3535, 3582, 3589, 3604, 3797, 3798, 3799, 3801, 3807, 3908, 3912, 3990, 3992, 4127, 4128, 4129, 4213, 4219, 4251, 4265, 4309, 4382, 4383, 4528, 4571, 4631, 4654, 4657, 5139, 5270, 5313, 5315, 5600, 5601, 5686, 5715, 5716, 5740, 5742, 5792, 5850]
chunk_over_5_loss.sort(reverse=True)
step, total_loss=0,0
for item in chunk_over_5_loss:
    del pre_train_tensors_list[item]

print("an item from the pretrainlist crabplover", pre_train_tensors_list[1].size())
seagul_sized_pre_train_tensors_list = [chunk for tensor in pre_train_tensors_list for chunk in torch.chunk(tensor, 10, dim=0)] #restructure the list to match the durations of the seagull dataset
print("an item from the pretrainlist seagull", seagul_sized_pre_train_tensors_list[1].size())
check_for_nans(torch.stack(seagul_sized_pre_train_tensors_list), "seagul_sized_pre_train_tensors_list")
random.seed(1)
random.shuffle(seagul_sized_pre_train_tensors_list)
pretrain_epochs = 20
masking_percentage = 0.5
for epoch in range(pretrain_epochs):
    errors = []
    total_loss = 0            
    for tensor_20hz in seagul_sized_pre_train_tensors_list:
        tensor_20hz = tensor_20hz#.to(device)
        masked_segments, binary_mask, original_segments = partition_and_mask(tensor_20hz,segment_size=2,mask_percentage=0.7) #think about the segment_size
        masked_segments = masked_segments.to(torch.float32)#.to(device)#.float()
        original_segments = original_segments.to(torch.float32)#.to(device)#.float()
        encoded_segments = pretrained_encoder(masked_segments,binary_mask)#.to(device)
        binary_mask_expanded = binary_mask.unsqueeze(-1).repeat(1, 1, 1, decoder.num_heads).view(binary_mask.size(0), binary_mask.size(1), -1)#.to(device)
        boolean_mask_expanded = binary_mask_expanded.to(torch.bool)#.to(device) #turn mask into boolean mask 
        reconstructed_data = decoder(encoded_segments, boolean_mask_expanded)#.to(device)
        reconstructed_data = reconstructed_data.to(torch.float32)#.to(device)
        reconstruction_loss = nn.MSELoss()#.to(device) #or mean average error: reconstruction_loss = nn.L1Loss() #MAE are scalable learners uses MSEloss too

        loss = reconstruction_loss(reconstructed_data, original_segments)
        
        # Backpropagate and update weights
        optimizer = optim.Adam(list(pretrained_encoder.parameters()) + list(decoder.parameters()), lr=0.001,weight_decay=1e-5) #MAE are scalable learners uses Adam, weight_decay is regularisation
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(pretrained_encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()
        step+=1
        total_loss += loss.item()
        
        writer.add_scalar('Loss/Reconstruction_pre', loss.item(), step) 

    writer.add_scalar('Loss/Average_loss_pre', total_loss/step, epoch)



state_dict = pretrained_encoder.state_dict()
for name, param in state_dict.items():
    if torch.isnan(param).any():
        print(f"NaN detected in state dict of {name}")



size_contribution, offset_contribution, downsample_factor = 0.2, 1, 4
untrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.1, sequence_length=20)

#save_model(pretrained_encoder, optimizer, epoch=20, mskpct=0.5)
#for 20 epoch pretraining and mskpct of 0.5; the save model function epoch was placed at 0 and mskpct too. 
# Usage

loss_tracker = []

model = TimeSeriesCenterNet(pretrained_encoder, num_classes=num_classes,downsampling_factor=downsample_factor, sequence_length=sequence_length) #unspecified, out of sight, interaction and shit are removed (shake too)
model = model#.to(device)
optimizer = optim.Adam(list(model.parameters()), lr=0.001,weight_decay=1e-5)
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
fold_size = int(train_data_size/num_folds)

trainingstep = 0
validationstep = 0

#given the larger batchsize, we switch to GPU
print(f"batch_size{batch_size}")
print(f"num_fols{num_folds}")
print(f"batches within fold{batches_within_fold}")
for fold, (train_ids, test_ids) in enumerate(kfold.split(trainingset)): #trainingset already stratified
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids) #splits into traindata
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids) #splits into testdata
    train_loader = DataLoader(trainingset, batch_size=batch_size, sampler=train_subsampler, collate_fn=custom_collate_fn)#, pin_memory=True)
    validation_loader = DataLoader(trainingset, batch_size=batch_size, sampler=test_subsampler, collate_fn=custom_collate_fn)#, pin_memory=True)
    print("length of train_loader",len(train_loader)) # ==50
    print("length of validation_loader", len(validation_loader))# ==10
    for epoch in range(10):  
        model.train()
        total_predictions_aa = []
        total_predictions_sma = []
        total_predictions_gs = []
        total_labels = []
        for features, labels , durations, keypoints, labels_list in train_loader: 
            #features, labels = features.to(device), labels.to(device)
            the_size_of_features = features.size()    
            optimizer.zero_grad()
            heatmap_target = generate_heatmaps(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=num_classes, durations_batch= durations, downsample_factor=4)#.to(device)
            sizemap_target = generate_size_maps(sequence_length=sequence_length ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)#.to(device)
            offset_target = generate_offset_map(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)#.to(device)
            mask = torch.ones(features.size(), dtype=torch.bool)#.to(device)
            output = model(features,mask,downsample_factor) #adding mask to match the encoder architecture
            heatmap_prediction, size_prediction, offset_prediction = output 
            size_prediction = size_prediction.squeeze(1)
            offset_prediction = offset_prediction.squeeze(1)

            heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target, weight_tensor=labels_for_refrence ,class_weights_activated=True) #manual instantation of focalloss
            l1_loss_size = l1_loss(size_prediction, sizemap_target)
            l1_loss_offset = l1_loss(offset_prediction, offset_target) 
            check_for_nans(heatmap_prediction, 'heatmap_prediction')
            check_for_nans(size_prediction, 'size_prediction')
            check_for_nans(offset_prediction,'offset_prediction')

            

            total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset
            #loss_tracker = copy.deepcopy(total_loss)
            #loss_tracker.append(total_loss.detach().clone())
            #print("this is the total_loss:",total_loss)
            total_loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN detected in gradients of {name} before clipping")
                    #print(f"Gradients of {name} - min: {param.min().item()}, max: {param.max().item()}")


            peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (window_size/downsample_factor), prominence_factor=0.75)               
            combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
            filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

            ###reconstruct_timelines_comparisons
            total_labels.append(labels)  
            total_labels_tensor = torch.stack(total_labels, dim=0).flatten()

            reconstructed_timelines_aa = reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length)
            reconstructed_timelines_gs = reconstruct_timelines_gaussian_support(filtered_peaks, sequence_length)            
            reconstructed_timelines_sma = reconstruct_timelines_start_max_activation(filtered_peaks,sequence_length)

            total_predictions_aa.append(reconstructed_timelines_aa)
            total_predictions_gs.append(reconstructed_timelines_gs)
            total_predictions_sma.append(reconstructed_timelines_sma)
                    
            total_predictions_aa_tensor = torch.stack(total_predictions_aa, dim=0).flatten()
            total_predictions_gs_tensor = torch.stack(total_predictions_gs, dim=0).flatten()
            total_predictions_sma_tensor = torch.stack(total_predictions_sma, dim=0).flatten()

            average_accuracy_aa, average_accuracy_gs, average_accuracy_sma  = MulticlassAccuracy(), MulticlassAccuracy(), MulticlassAccuracy()
            average_recall_aa, average_recall_gs, average_recall_sma  = MulticlassRecall(), MulticlassRecall(), MulticlassRecall()

            average_accuracy_aa.update(total_predictions_aa_tensor, total_labels_tensor)
            average_accuracy_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            average_accuracy_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            
            average_recall_aa.update(total_predictions_aa_tensor, total_labels_tensor)
            average_recall_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            average_recall_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            ###


            clip_and_threshold_gradients(model.encoder.parameters(), clip_value=1.0, min_value=1e-10)
            #clipper = ClipConstraint(22.3)
            #clipper(model)
            optimizer.step()

            writer.add_scalars('Training/Accuracy', {
                'last activation average accuracy' : average_accuracy_aa.compute(),
                'start maximum activation average accuracy' : average_accuracy_sma.compute(),
                'gaussian support average accuracy' : average_accuracy_gs.compute() 
            }, trainingstep)

            writer.add_scalars('Training/recall', {
                'last activation average recall' : average_recall_aa.compute(),
                'start maximum activation average recall' : average_recall_sma.compute(),
                'gaussian support average recall' : average_recall_gs.compute() 
            }, trainingstep)
            
            writer.add_scalar('Training/size_loss', l1_loss_size.item(), trainingstep)#epoch * len(train_loader) + fold)
            writer.add_scalar('Training/heatmap_loss', heatmap_loss.item(), trainingstep)#epoch * len(train_loader) + fold)
            writer.add_scalar('Training/offset_loss', l1_loss_offset.item(), trainingstep)# epoch * len(train_loader) + fold)          
            writer.add_scalar('Training/total_loss', total_loss.item(), trainingstep)# epoch * len(train_loader) + fold) #item()
        trainingstep += 1

        # Validate
        model.eval()
        with torch.no_grad():
            validation_loss, heatmaploss_tot, l1_loss_offset_tot, l1_loss_size_tot = 0,0,0,0
            total_predictions_val_aa = []
            total_labels_val= [] ####make a tensor with all the labels and one with all the predictions then try average recall and precision
            for features, labels , durations, keypoints, labels_list in validation_loader:
                #features = features.to(device)
                #labels = labels.to(device)
                heatmap_target = generate_heatmaps(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=num_classes, durations_batch= durations, downsample_factor=4)#.to(device)
                sizemap_target = generate_size_maps(sequence_length=sequence_length ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)#.to(device)
                offset_target = generate_offset_map(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)#.to(device)
                mask = torch.ones(features.size(), dtype=torch.bool)#, device=device)
                output = model(features,mask, downsample_factor) #adding mask to match the encoder architecture                
                heatmap_prediction, size_prediction, offset_prediction = output
                size_prediction = size_prediction.squeeze(1)
                offset_prediction = offset_prediction.squeeze(1) 
                heatmaploss = manual_loss_v2(heatmap_prediction, heatmap_target,weight_tensor=labels_for_refrence ,class_weights_activated=True)#, class_weights_activated=True)                

                peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (window_size/downsample_factor), prominence_factor=0.75)
                                
                combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
                filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

                total_predictions_val_aa.append(reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length))
                total_labels_val.append(labels)
                total_predictions_val_aa_tensor = torch.stack(total_predictions_val_aa, dim=0).flatten()#.to(device)
                total_labels_val_tensor = torch.stack(total_labels_val, dim=0).flatten()#.to(device)
                accuracy_per_class = MulticlassAccuracy(average=None, num_classes=num_classes)
                average_accuracy = MulticlassAccuracy()
                recall_per_class = MulticlassRecall(average=None, num_classes=num_classes)
                average_recall = MulticlassRecall()

                accuracy_per_class.update(total_predictions_val_aa_tensor, total_labels_val_tensor)
                average_accuracy.update(total_predictions_val_aa_tensor, total_labels_val_tensor)
                recall_per_class.update(total_predictions_val_aa_tensor, total_labels_val_tensor)
                average_recall.update(total_predictions_val_aa_tensor, total_labels_val_tensor)
                print("accuracy per class:", accuracy_per_class.compute())
                print("recall per class:", recall_per_class.compute())
                
                writer.add_scalar('Validation/average accuracy', average_accuracy.compute(), fold)
                writer.add_scalar('Validation/average recall', average_recall.compute(), fold)            

                l1_loss_size = l1_loss(size_prediction, sizemap_target)
                l1_loss_offset = l1_loss(offset_prediction, offset_target)            
                total_loss = heatmaploss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

                heatmaploss_tot+= heatmaploss
                validation_loss+= total_loss
                l1_loss_size_tot += l1_loss_size
                l1_loss_offset_tot+= l1_loss_offset
                
                #writer.add_scalar('validationcosine_similarity', cosine_similarity,epoch * len(validation_loader) + fold )
                writer.add_scalar('Validation/size_loss', l1_loss_size, validationstep)#epoch * len(validation_loader) + fold)
                writer.add_scalar('Validation/heatmap_loss', heatmaploss, validationstep)# epoch * len(validation_loader) + fold)
                writer.add_scalar('Validation/offset_loss', l1_loss_offset, validationstep)# epoch * len(train_loader) + fold)          
                writer.add_scalar('Validation/total_loss', total_loss, validationstep)# epoch * len(validation_loader) + fold)
            validationstep+=1

        print(f'Fold {fold}, Epoch {epoch}, Size Loss: {l1_loss_size_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Heatmap Loss: {heatmaploss_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, offset Loss: {l1_loss_offset_tot / len(validation_loader)}')
        print(f'Fold {fold}, Epoch {epoch}, Total Validation Loss: {validation_loss / len(validation_loader)}')


#now there will be a segment where the batchsize will be a lot larger than earlier. Resulting in the posssibility of the kernel dying
#where CUDA created overload in the previous segment, it will prove usefull in this segment
#model.to(device)


#now I want to see the evaluation metrics accros the entire training_dataset:
train_loader = DataLoader(trainingset, batch_size=len(trainingset), collate_fn=custom_collate_fn ,pin_memory=True)

#now I want to see the evaluation metrics accros the entire training_dataset:

"""
for features, labels , durations, keypoints, labels_list in train_loader:  
    features = features.to(device)
    labels = labels.to(device) 
    heatmap_target = generate_heatmaps(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=num_classes, durations_batch= durations, downsample_factor=4).to(device)
    sizemap_target = generate_size_maps(sequence_length=sequence_length ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4).to(device)
    offset_target = generate_offset_map(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4).to(device)
    mask = torch.ones(features.size(), dtype=torch.bool).to(device)
    output = model(features,mask,4) #manual filling the downsamplesize
    heatmap_prediction, size_prediction, offset_prediction = output 
    size_prediction = size_prediction.squeeze(1)
    offset_prediction = offset_prediction.squeeze(1)

    heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target,weight_tensor=labels_for_refrence , class_weights_activated=False, device=True).to(device) # ,class_weights_activated=True

    l1_loss_size = l1_loss(size_prediction, sizemap_target)
    l1_loss_offset = l1_loss(offset_prediction, offset_target) 
    total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

    peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (window_size/downsample_factor), prominence_factor=0.75)               
    combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
    filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

    ###reconstruct_timelines_comparisons
    total_labels_tensor = labels.flatten().to(device)
    labels_sklearn = total_labels_tensor.cpu().numpy().flatten()
    
    reconstructed_timelines_aa = reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length).flatten().to(device)
    reconstructed_timelines_gs = reconstruct_timelines_gaussian_support(filtered_peaks, sequence_length).flatten().to(device)           
    reconstructed_timelines_sma = reconstruct_timelines_start_max_activation(filtered_peaks,sequence_length).flatten().to(device)

    timelines_aa_sklearn, timelines_gs_sklearn, timelines_sma_sklearn = reconstructed_timelines_aa.cpu().numpy().flatten(), reconstructed_timelines_gs.cpu().numpy().flatten(), reconstructed_timelines_sma.cpu().numpy().flatten()

    accuracy_aa, accuracy_gs, accuracy_sma  = MulticlassAccuracy(), MulticlassAccuracy(), MulticlassAccuracy()
    recall_aa, recall_gs, recall_sma  = MulticlassRecall(), MulticlassRecall(), MulticlassRecall()
    curve_aa, curve_gs, curve_sma = MulticlassPrecisionRecallCurve(num_classes=num_classes), MulticlassPrecisionRecallCurve(num_classes=num_classes), MulticlassPrecisionRecallCurve(num_classes=num_classes)
    
    precision_aa, precision_gs, precision_sma = precision_score(timelines_aa_sklearn, labels_sklearn, average=None, zero_division=0), precision_score(timelines_gs_sklearn, labels_sklearn, average=None, zero_division=0), precision_score(timelines_sma_sklearn, labels_sklearn, average=None, zero_division=0) #change to gpu if better suiting, cpu ran better in my case


    f1_score_aa , f1_score_gs, f1_score_sma= f1_score(reconstructed_timelines_aa.cpu().numpy(), total_labels_tensor.cpu().numpy(), average=None), f1_score(reconstructed_timelines_gs.cpu().numpy(), total_labels_tensor.cpu().numpy(), average=None), f1_score(reconstructed_timelines_sma.cpu().numpy(), total_labels_tensor.cpu().numpy(), average=None)

    accuracy_aa.update(reconstructed_timelines_aa, total_labels_tensor)
    accuracy_gs.update(reconstructed_timelines_gs, total_labels_tensor)
    accuracy_sma.update(reconstructed_timelines_sma, total_labels_tensor)
            
    recall_aa.update(reconstructed_timelines_aa, total_labels_tensor)
    recall_gs.update(reconstructed_timelines_gs, total_labels_tensor)
    recall_sma.update(reconstructed_timelines_sma, total_labels_tensor)
    

    accuracy_per_class_aa = MulticlassAccuracy(average=None, num_classes=num_classes) #usefull during trainig to see which classes are underrepresented (14 is absent)
    accuracy_per_class_aa.update(reconstructed_timelines_aa, total_labels_tensor)

    recall_per_class_aa = MulticlassRecall(average=None, num_classes=num_classes)
    recall_per_class_aa.update(reconstructed_timelines_aa, total_labels_tensor)

    accuracy_per_class_gs = MulticlassAccuracy(average=None, num_classes=num_classes) 
    accuracy_per_class_gs.update(reconstructed_timelines_gs, total_labels_tensor)

    recall_per_class_gs = MulticlassRecall(average=None, num_classes=num_classes)
    recall_per_class_gs.update(reconstructed_timelines_gs, total_labels_tensor)

    accuracy_per_class_sma = MulticlassAccuracy(average=None, num_classes=num_classes)
    accuracy_per_class_sma.update(reconstructed_timelines_sma, total_labels_tensor)

    recall_per_class_sma = MulticlassRecall(average=None, num_classes=num_classes)
    recall_per_class_sma.update(reconstructed_timelines_sma, total_labels_tensor)

    fig_aa_cm = plot_confusion_matrix(labels_sklearn, timelines_aa_sklearn, 'Normalized Confusion Matrix (Last Activation)')
    fig_gs_cm = plot_confusion_matrix(labels_sklearn, timelines_gs_sklearn, 'Normalized Confusion Matrix (Gaussian Support)')
    fig_sma_cm = plot_confusion_matrix(labels_sklearn, timelines_sma_sklearn, 'Normalized Confusion Matrix (Start Max Activation)')

    fig_aa_cm = plot_confusion_matrix(labels_sklearn, timelines_aa_sklearn,  'Confusion Matrix (Last Activation)', normalize=True)
    fig_gs_cm = plot_confusion_matrix(labels_sklearn, timelines_gs_sklearn, 'Confusion Matrix (Gaussian Support)' , normalize=True)
    fig_sma_cm = plot_confusion_matrix(labels_sklearn, timelines_sma_sklearn,  'Confusion Matrix (Start Max Activation)', normalize=True)


    # Save figures to TensorBoard
    writer.add_figure('Train/Confusion_Matrix_Last_Activation', fig_aa_cm)
    writer.add_figure('Train/Confusion_Matrix_Gaussian_Support', fig_gs_cm)
    writer.add_figure('Train/Confusion_Matrix_Start_Max_Activation', fig_sma_cm)


    precision_dict = {
        'Last Activation': torch.tensor(precision_aa),
        'Gaussian Support': torch.tensor(precision_gs),
        'Start Max Activation': torch.tensor(precision_sma)
    }

    recall_dict = {
        'Last Activation': recall_per_class_aa.compute(),
        'Gaussian Support': recall_per_class_gs.compute(),
        'Start Max Activation': recall_per_class_sma.compute()
    }

    f1_dict = {
        'Last Activation': torch.tensor(f1_score_aa),
        'Gaussian Support': torch.tensor(f1_score_gs),
        'Start Max Activation': torch.tensor(f1_score_sma)
    }

    plot_bar_chart(precision_dict, 'Precision', writer, 0, num_classes, test_train_val='Train')
    plot_bar_chart(recall_dict, 'Recall', writer, 0, num_classes,  test_train_val='Train')
    plot_bar_chart(f1_dict, 'F1-Score', writer, 0, num_classes,  test_train_val='Train')



    print("accuracy per class:_aa", accuracy_per_class_aa.compute())
    print("recall per class:_aa", recall_per_class_aa.compute())

    print("accuracy per class gs", accuracy_per_class_gs.compute())
    print("recall per class gs", recall_per_class_gs.compute())

    print("accuracy per class sma:",accuracy_per_class_sma.compute())
    print("recall per class sma", recall_per_class_sma.compute())

    print("f1_score_per_class_aa", f1_score_aa)
    print("f1_score_per_class_gs", f1_score_gs)
    print("f1_score_per_class_sma", f1_score_sma)

    print("precision_per_class_aa", precision_aa)
    print("precision_per_class_gs", precision_gs)
    print("precision_per_class_sma", precision_sma)


    plt.tight_layout()
    
    writer.add_scalars('train/Accuracy', {
        'last activation average accuracy' : accuracy_aa.compute(),
        'start maximum activation average accuracy' : accuracy_sma.compute(),
        'gaussian support average accuracy' : accuracy_gs.compute() 
    }, 0)

    writer.add_scalars('train/recall', {
        'last activation average recall' : recall_aa.compute(),
        'start maximum activation average recall' : recall_sma.compute(),
        'gaussian support average recall' : recall_gs.compute() 
    }, 0)

"""
test_loader = DataLoader(testset, batch_size=len(testset), collate_fn=custom_collate_fn)# ,pin_memory=True)

for features, labels , durations, keypoints, labels_list in test_loader:  
    #features = features.to(device)
    #labels = labels.to(device) 
    heatmap_target = generate_heatmaps(sequence_length=sequence_length, batch_size=test_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=num_classes, durations_batch= durations, downsample_factor=4)#.to(device)
    sizemap_target = generate_size_maps(sequence_length=sequence_length ,batch_size=test_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)#.to(device)
    offset_target = generate_offset_map(sequence_length=sequence_length, batch_size=test_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)#.to(device)
    mask = torch.ones(features.size(), dtype=torch.bool)#.to(device)
    output = model(features,mask,4) #manual filling the downsamplesize
    heatmap_prediction, size_prediction, offset_prediction = output 
    size_prediction = size_prediction.squeeze(1)
    offset_prediction = offset_prediction.squeeze(1)

    heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target ,weight_tensor=labels_for_refrence , class_weights_activated=True)#, device=True)#.to(device) # ,class_weights_activated=True

    l1_loss_size = l1_loss(size_prediction, sizemap_target)
    l1_loss_offset = l1_loss(offset_prediction, offset_target) 
    total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

    peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (window_size/downsample_factor), prominence_factor=0.75)               
    combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
    filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

    ###reconstruct_timelines_comparisons
    total_labels_tensor = labels.flatten()#.to(device)
    labels_sklearn = total_labels_tensor.cpu().numpy().flatten()
    
    reconstructed_timelines_aa = reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length).flatten()#.to(device)
    reconstructed_timelines_gs = reconstruct_timelines_gaussian_support(filtered_peaks, sequence_length).flatten()#.to(device)           
    reconstructed_timelines_sma = reconstruct_timelines_start_max_activation(filtered_peaks,sequence_length).flatten()#.to(device)

    timelines_aa_sklearn, timelines_gs_sklearn, timelines_sma_sklearn = reconstructed_timelines_aa.cpu().numpy().flatten(), reconstructed_timelines_gs.cpu().numpy().flatten(), reconstructed_timelines_sma.cpu().numpy().flatten()
    #The macro's don't work, extract from confusion matrix
    accuracy_aa, accuracy_gs, accuracy_sma  = MulticlassAccuracy(), MulticlassAccuracy(), MulticlassAccuracy()
 
    recall_aa, recall_gs, recall_sma  = MulticlassRecall(), MulticlassRecall(), MulticlassRecall()
    curve_aa, curve_gs, curve_sma = MulticlassPrecisionRecallCurve(num_classes=num_classes), MulticlassPrecisionRecallCurve(num_classes=num_classes), MulticlassPrecisionRecallCurve(num_classes=num_classes)
    
    precision_aa, precision_gs, precision_sma = precision_score(timelines_aa_sklearn, labels_sklearn, average=None, zero_division=0), precision_score(timelines_gs_sklearn, labels_sklearn, average=None, zero_division=0), precision_score(timelines_sma_sklearn, labels_sklearn, average=None, zero_division=0) #change to gpu if better suiting, cpu ran better in my case 
    f1_score_aa , f1_score_gs, f1_score_sma= f1_score(reconstructed_timelines_aa.cpu().numpy(), total_labels_tensor.cpu().numpy(), average=None), f1_score(reconstructed_timelines_gs.cpu().numpy(), total_labels_tensor.cpu().numpy(), average=None), f1_score(reconstructed_timelines_sma.cpu().numpy(), total_labels_tensor.cpu().numpy(), average=None)

    accuracy_aa.update(reconstructed_timelines_aa, total_labels_tensor)
    accuracy_gs.update(reconstructed_timelines_gs, total_labels_tensor)
    accuracy_sma.update(reconstructed_timelines_sma, total_labels_tensor)

    recall_aa.update(reconstructed_timelines_aa, total_labels_tensor)
    recall_gs.update(reconstructed_timelines_gs, total_labels_tensor)
    recall_sma.update(reconstructed_timelines_sma, total_labels_tensor)

    accuracy_per_class_aa = MulticlassAccuracy(average=None, num_classes=num_classes) #usefull during trainig to see which classes are underrepresented (14 is absent)
    accuracy_per_class_aa.update(reconstructed_timelines_aa, total_labels_tensor)

    recall_per_class_aa = MulticlassRecall(average=None, num_classes=num_classes)
    recall_per_class_aa.update(reconstructed_timelines_aa, total_labels_tensor)

    accuracy_per_class_gs = MulticlassAccuracy(average=None, num_classes=num_classes) 
    accuracy_per_class_gs.update(reconstructed_timelines_gs, total_labels_tensor)

    recall_per_class_gs = MulticlassRecall(average=None, num_classes=num_classes)
    recall_per_class_gs.update(reconstructed_timelines_gs, total_labels_tensor)

    accuracy_per_class_sma = MulticlassAccuracy(average=None, num_classes=num_classes)
    accuracy_per_class_sma.update(reconstructed_timelines_sma, total_labels_tensor)

    recall_per_class_sma = MulticlassRecall(average=None, num_classes=num_classes)
    recall_per_class_sma.update(reconstructed_timelines_sma, total_labels_tensor)

    fig_aa_cm = plot_confusion_matrix(labels_sklearn, timelines_aa_sklearn, 'Normalized Confusion Matrix (Last Activation)')
    fig_gs_cm = plot_confusion_matrix(labels_sklearn, timelines_gs_sklearn, 'Normalized Confusion Matrix (Gaussian Support)')
    fig_sma_cm = plot_confusion_matrix(labels_sklearn, timelines_sma_sklearn, 'Normalized Confusion Matrix (Start Max Activation)')

    # Save figures to TensorBoard
    writer.add_figure('Test/Confusion_Matrix_Last_Activation', fig_aa_cm)
    writer.add_figure('Test/Confusion_Matrix_Gaussian_Support', fig_gs_cm)
    writer.add_figure('Test/Confusion_Matrix_Start_Max_Activation', fig_sma_cm)


    precision_dict = {
        'Last Activation': torch.tensor(precision_aa),
        'Gaussian Support': torch.tensor(precision_gs),
        'Start Max Activation': torch.tensor(precision_sma)
    }

    recall_dict = {
        'Last Activation': recall_per_class_aa.compute(),
        'Gaussian Support': recall_per_class_gs.compute(),
        'Start Max Activation': recall_per_class_sma.compute()
    }

    f1_dict = {
        'Last Activation': torch.tensor(f1_score_aa),
        'Gaussian Support': torch.tensor(f1_score_gs),
        'Start Max Activation': torch.tensor(f1_score_sma)
    }

    plot_bar_chart(precision_dict, 'Precision', writer, 0, num_classes, test_train_val='Test')
    plot_bar_chart(recall_dict, 'Recall', writer, 0, num_classes,  test_train_val='Test')
    plot_bar_chart(f1_dict, 'F1-Score', writer, 0, num_classes,  test_train_val='Test')

    print("accuracy per class:_aa", accuracy_per_class_aa.compute())
    print("recall per class:_aa", recall_per_class_aa.compute())

    print("accuracy per class gs", accuracy_per_class_gs.compute())
    print("recall per class gs", recall_per_class_gs.compute())

    print("accuracy per class sma:",accuracy_per_class_sma.compute())
    print("recall per class sma", recall_per_class_sma.compute())

    print("f1_score_per_class_aa", f1_score_aa)
    print("f1_score_per_class_gs", f1_score_gs)
    print("f1_score_per_class_sma", f1_score_sma)

    print("precision_per_class_aa", precision_aa)
    print("precision_per_class_gs", precision_gs)
    print("precision_per_class_sma", precision_sma)


    plt.tight_layout()
    
    writer.add_scalars('Test/Accuracy', {
        'last activation average accuracy' : accuracy_aa.compute(),
        'start maximum activation average accuracy' : accuracy_sma.compute(),
        'gaussian support average accuracy' : accuracy_gs.compute() 
    }, 0)

    writer.add_scalars('Test/recall', {
        'last activation average recall' : recall_aa.compute(),
        'start maximum activation average recall' : recall_sma.compute(),
        'gaussian support average recall' : recall_gs.compute() 
    }, 0)

    writer.add_scalar('Test/size_loss', l1_loss_size.item(), 0)#epoch * len(train_loader) + fold)
    writer.add_scalar('Test/heatmap_loss', heatmap_loss.item(), 0)#epoch * len(train_loader) + fold)
    writer.add_scalar('Test/offset_loss', l1_loss_offset.item(),0)# epoch * len(train_loader) + fold)          
    writer.add_scalar('Test/total_loss', total_loss.item(), 0)# epoch * len(train_loader) + fold) #item()

writer.close()
