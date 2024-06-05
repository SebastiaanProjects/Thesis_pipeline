from MAE_utils import TimeSeriesMAEEncoder, TimeSeriesMAEDecoder
from CenterNet_utils import TimeSeriesCenterNet, generate_heatmaps, generate_offset_map, generate_size_maps, manual_loss_v2, l1_loss, visualize_heatmap, visualize_size_map, visualize_offset_map, evaluate_adaptive_peak_extraction, neighbouring_peaks_sort, iou_based_peak_suppression, iou_1d, reconstruct_timelines_ascending_activation, combine_peaks_with_maps, reconstruct_timelines_gaussian_support, reconstruct_timelines_start_max_activation
from Data_extraction_utils import custom_collate_fn, TimeSeriesDataset

from data_composer import trainingset, testset, labels_for_refrence, sequence_length, num_classes, num_folds, train_data_size, batch_size#, pre_train_tensors_list, 

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
from sklearn.metrics import precision_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set()
writer = SummaryWriter()

#trainingset = TimeSeriesDataset(original_features_train, train_labels_indexed, durations_all_train,  keypoints_all_train, indexed_labels_list_train)

unique_labels_len = len(np.unique(labels_for_refrence))
size_contribution, offset_contribution, downsample_factor = 0.2, 1, 4

untrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.1)


model = TimeSeriesCenterNet(untrained_encoder, num_classes=num_classes,downsampling_factor=downsample_factor, sequence_length=sequence_length) #unspecified, out of sight, interaction and shit are removed (shake too)
model = model#.to(device)
optimizer = optim.Adam(list(model.parameters()), lr=0.001,weight_decay=1e-5)

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
fold_size = int(train_data_size/num_folds)

trainingstep = 0
validationstep = 0

print(num_folds)
print(batch_size)

for fold, (train_ids, test_ids) in enumerate(kfold.split(trainingset)): #trainingset already stratified
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids) #splits into traindata
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids) #splits into testdata
    # Create DataLoaders for train and validation dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader = DataLoader(trainingset, batch_size=batch_size, sampler=train_subsampler, collate_fn=custom_collate_fn)#, pin_memory=True )
    validation_loader = DataLoader(trainingset, batch_size=batch_size, sampler=test_subsampler, collate_fn=custom_collate_fn)#, pin_memory=True)
    print("length of train_loader",len(train_loader))
    print("length of validation_loader", len(validation_loader))
    for epoch in range(0):  
        model.train()
        total_predictions_aa = []
        total_predictions_sma = []
        total_predictions_gs = []
        total_labels = []
        for features, labels , durations, keypoints, labels_list in train_loader:          
            optimizer.zero_grad()
            heatmap_target = generate_heatmaps(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4)
            sizemap_target = generate_size_maps(sequence_length=sequence_length ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)
            offset_target = generate_offset_map(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
            mask = torch.ones(features.size(), dtype=torch.bool)#.to(device)
            output = model(features,mask,downsample_factor) #adding mask to match the encoder architecture
            heatmap_prediction, size_prediction, offset_prediction = output 
            size_prediction = size_prediction.squeeze(1)
            offset_prediction = offset_prediction.squeeze(1)

            heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target) #manual instantation of focalloss


            l1_loss_size = l1_loss(size_prediction, sizemap_target)
            l1_loss_offset = l1_loss(offset_prediction, offset_target) 
            total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

            total_loss.backward()
            peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (40/downsample_factor), prominence_factor=0.75)               
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

            #accuracy_per_class = MulticlassAccuracy(average=None, num_classes=num_classes) #usefull during trainig to see which classes are underrepresented (14 is absent)
            #accuracy_per_class.update(total_predictions_aa_tensor, total_labels_tensor)
            #recall_per_class = MulticlassRecall(average=None, num_classes=num_classes)
            #recall_per_class.update(total_predictions_aa_tensor, total_labels_tensor)
            #print("accuracy per class:", accuracy_per_class.compute())
            #print("recall per class:", recall_per_class.compute())

            average_accuracy_aa, average_accuracy_gs, average_accuracy_sma  = MulticlassAccuracy(), MulticlassAccuracy(), MulticlassAccuracy()
            average_recall_aa, average_recall_gs, average_recall_sma  = MulticlassRecall(), MulticlassRecall(), MulticlassRecall()
            average_precision_aa, average_precision_gs, average_precision_sma = precision_score(total_predictions_aa_tensor.cpu().numpy().flatten(), total_labels_tensor.cpu().numpy().flatten(), average='macro', zero_division=0), precision_score(total_predictions_gs_tensor.cpu().numpy().flatten(), total_labels_tensor.cpu().numpy().flatten(), average='macro', zero_division=0), precision_score(total_predictions_sma_tensor.cpu().numpy().flatten(), total_labels_tensor.cpu().numpy().flatten(), average='macro', zero_division=0) #change to gpu if better suiting, cpu ran better in my case
            
            average_accuracy_aa.update(total_predictions_aa_tensor, total_labels_tensor)
            average_accuracy_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            average_accuracy_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            
            average_recall_aa.update(total_predictions_aa_tensor, total_labels_tensor)
            average_recall_gs.update(total_predictions_gs_tensor, total_labels_tensor)
            average_recall_sma.update(total_predictions_sma_tensor, total_labels_tensor)
            ###

            optimizer.step()

            ###
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

            writer.add_scalars('Training/precision', {
                'last activation average recall' : average_precision_aa,
                'start maximum activation average recall' : average_precision_sma,
                'gaussian support average recall' : average_precision_gs 
            }, trainingstep)

            ###
            
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
                heatmap_target = generate_heatmaps(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4)
                sizemap_target = generate_size_maps(sequence_length=sequence_length ,batch_size=train_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4)
                offset_target = generate_offset_map(sequence_length=sequence_length, batch_size=train_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4)
                mask = torch.ones(features.size(), dtype=torch.bool)#, device=device)
                output = model(features,mask, downsample_factor) #adding mask to match the encoder architecture
                
                heatmap_prediction, size_prediction, offset_prediction = output
                size_prediction = size_prediction.squeeze(1)
                offset_prediction = offset_prediction.squeeze(1) 
                heatmaploss = manual_loss_v2(heatmap_prediction, heatmap_target)                

                peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (40/downsample_factor), prominence_factor=0.75)
                                
                combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
                filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

                total_predictions_val_aa.append(reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length))
                total_labels_val.append(labels)
                total_predictions_val_aa_tensor = torch.stack(total_predictions_val_aa, dim=0).flatten()
                total_labels_val_tensor = torch.stack(total_labels_val, dim=0).flatten()
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
                
                writer.add_scalar('Validation/average precision', average_accuracy.compute(), fold)
                writer.add_scalar('Validation/average recall', average_recall.compute(), fold)

                if fold > 10:
                    torch.set_printoptions(profile="full")
                    print("combined peaks[0]", combined_peaks[0])  
                    print("filtered peaks[0]", filtered_peaks[0])
                    print("reconstructed_timelines_aa[0]",reconstructed_timelines_aa[0])
                    print("labels[0]", labels[0])                   

                    torch.set_printoptions(profile="default") # reset               

                l1_loss_size = l1_loss(size_prediction, sizemap_target)
                l1_loss_offset = l1_loss(offset_prediction, offset_target)            
                total_loss = heatmaploss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

                def correlation_coefficient(tensor_a, tensor_b):
                    a_mean = tensor_a - tensor_a.mean(dim=1, keepdim=True)
                    b_mean = tensor_b - tensor_b.mean(dim=1, keepdim=True)
                    numerator = (a_mean * b_mean).sum(dim=1)
                    denominator = torch.sqrt((a_mean ** 2).sum(dim=1) * (b_mean ** 2).sum(dim=1))
                    return numerator / denominator


                #correlation = correlation_coefficient(labels, reconstructed_timelines_aa)

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

    torch.set_printoptions(profile="full")  
    #visualize_heatmap(heatmap_target, heatmap_prediction)
    #visualize_size_map(sizemap_target, size_prediction)
    #visualize_offset_map(offset_target, offset_prediction)
    torch.set_printoptions(profile="default") # reset 


#now there will be a segment where the batchsize will be a lot larger than earlier. Resulting in the posssibility of the kernel dying
#where CUDA created overload in the previous segment, it will prove usefull in this segment
model.to(device)
test_loader = DataLoader(testset, batch_size=len(testset), collate_fn=custom_collate_fn ,pin_memory=True)

for features, labels , durations, keypoints, labels_list in test_loader:  
    features = features.to(device)
    labels = labels.to(device) 
    heatmap_target = generate_heatmaps(sequence_length=sequence_length, batch_size=test_loader.batch_size, keypoints_batch= keypoints, classes_batch= labels_list, num_classes=unique_labels_len, durations_batch= durations, downsample_factor=4).to(device)
    sizemap_target = generate_size_maps(sequence_length=sequence_length ,batch_size=test_loader.batch_size, keypoints_batch=keypoints, durations_batch= durations, downsample_factor=4).to(device)
    offset_target = generate_offset_map(sequence_length=sequence_length, batch_size=test_loader.batch_size, keypoints_batch=keypoints, downsample_factor=4).to(device)
    mask = torch.ones(features.size(), dtype=torch.bool).to(device)
    output = model(features,mask,4) #manual filling the downsamplesize
    heatmap_prediction, size_prediction, offset_prediction = output 
    size_prediction = size_prediction.squeeze(1)
    offset_prediction = offset_prediction.squeeze(1)

    heatmap_loss = manual_loss_v2(heatmap_prediction, heatmap_target) #manual instantation of focalloss

    l1_loss_size = l1_loss(size_prediction, sizemap_target)
    l1_loss_offset = l1_loss(offset_prediction, offset_target) 
    total_loss = heatmap_loss + size_contribution * l1_loss_size + offset_contribution * l1_loss_offset

    peaks = evaluate_adaptive_peak_extraction(heatmap_prediction, window_size= (40/downsample_factor), prominence_factor=0.75)               
    combined_peaks = combine_peaks_with_maps(peaks, size_prediction, offset_prediction, downsample_factor)
    filtered_peaks = iou_based_peak_suppression(combined_peaks, iou_threshold=0.3)

    ###reconstruct_timelines_comparisons
    #total_labels.append(labels)  
    total_labels_tensor = labels.flatten().to(device)

    reconstructed_timelines_aa = reconstruct_timelines_ascending_activation(filtered_peaks, sequence_length).flatten()
    reconstructed_timelines_gs = reconstruct_timelines_gaussian_support(filtered_peaks, sequence_length).flatten()            
    reconstructed_timelines_sma = reconstruct_timelines_start_max_activation(filtered_peaks,sequence_length).flatten()

    accuracy_aa, accuracy_gs, accuracy_sma  = MulticlassAccuracy(), MulticlassAccuracy(), MulticlassAccuracy()
    recall_aa, recall_gs, recall_sma  = MulticlassRecall(), MulticlassRecall(), MulticlassRecall()
    precision_aa, precision_gs, precision_sma = precision_score(reconstructed_timelines_aa.device().numpy().flatten(), total_labels_tensor.device().numpy().flatten(), average='macro', zero_division=0), precision_score(reconstructed_timelines_gs.device().numpy().flatten(), total_labels_tensor.device().numpy().flatten(), average='macro', zero_division=0), precision_score(reconstructed_timelines_sma.cpu().numpy().flatten(), total_labels_tensor.cpu().numpy().flatten(), average='macro', zero_division=0) #change to gpu if better suiting, cpu ran better in my case
    f1_score_aa , f1_score_gs, f1_score_sma= f1_score(reconstructed_timelines_aa.numpy(), total_labels_tensor.numpy(), average='micro'), f1_score(reconstructed_timelines_gs.numpy(), total_labels_tensor.numpy(), average='micro'), f1_score(reconstructed_timelines_sma.numpy(), total_labels_tensor.numpy(), average='micro')


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
    


    print("accuracy per class:_aa", accuracy_per_class_aa.compute())
    print("recall per class:_aa", recall_per_class_aa.compute())

    print("accuracy per class gs", accuracy_per_class_gs.compute())
    print("recall per class gs", recall_per_class_gs.compute())

    print("accuracy per class sma:",accuracy_per_class_sma.compute())
    print("recall per class sma", recall_per_class_sma.compute())

 

    #add the other accuracies!!
    #add the confusion matrix!!
    #add the f1 score !!
    writer.add_scalars('Test/Accuracy', {
        'last activation average accuracy' : average_accuracy_aa.compute(),
        'start maximum activation average accuracy' : average_accuracy_sma.compute(),
        'gaussian support average accuracy' : average_accuracy_gs.compute() 
    }, 0)

    writer.add_scalars('Test/recall', {
        'last activation average recall' : average_recall_aa.compute(),
        'start maximum activation average recall' : average_recall_sma.compute(),
        'gaussian support average recall' : average_recall_gs.compute() 
    }, 0)

    writer.add_scalars('Test/precision', {
        'last activation average recall' : average_precision_aa,
        'start maximum activation average recall' : average_precision_sma,
        'gaussian support average recall' : average_precision_gs 
    }, 0)

    writer.add_histogram('Test/F1_Score_Histogram', {
        "last activation f1 score": f1_score_aa,
        "start maximum activation f1 score": f1_score_sma,
        "gaussian support f1 score": f1_score_gs}, 0)

    ###

    writer.add_scalar('Test/size_loss', l1_loss_size.item(), 0)#epoch * len(train_loader) + fold)
    writer.add_scalar('Test/heatmap_loss', heatmap_loss.item(), 0)#epoch * len(train_loader) + fold)
    writer.add_scalar('Test/offset_loss', l1_loss_offset.item(),0)# epoch * len(train_loader) + fold)          
    writer.add_scalar('Test/total_loss', total_loss.item(), 0)# epoch * len(train_loader) + fold) #item()

                   
                   






writer.close()




#need to get a working peak_extractor. 
# I am not convinced about the current working of the peak extraction
# also I want to tryout with the weighted tensor for focal_loss (manual_loss_v2)
# When I have the peak extraction working I can work on the real results like confusion matrices 
# recall and precision and what not
# the conversion from to original label then needs to work to but that is easy
# the test can also be done then