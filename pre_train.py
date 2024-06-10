from MAE_utils import TimeSeriesMAEDecoder, TimeSeriesMAEEncoder
from data_composer import pre_train_tensors_list
import torch
import torch.nn as nn
import torch.optim as optim
from Data_extraction_utils import partition_and_mask
from torch.utils.tensorboard import SummaryWriter 
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

#                                   4               64          16              3           0.1
pretrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.1)#.to(device) #64 may be higher. Just for decoder to decide. 
# use a parameter optimizer to decide, snellius  
decoder = TimeSeriesMAEDecoder(embed_dim=4, decoder_embed_dim=64, num_heads=16, num_layers=1, max_seq_length=10, dropout_rate=0.1)#.to(device) #num_mask_tokens=6. 0.1 like in literature
total_loss = 0
step = 0

errored_steps = []
print(len(pre_train_tensors_list))
chunk_over_5_loss =[46, 167, 250, 299, 301, 302, 340, 360, 392, 393, 394, 491, 492, 493, 611, 615, 695, 699, 757, 814, 821, 844, 871, 907, 908, 909, 910, 911, 912, 913, 918, 946, 988, 999, 1008, 1023, 1026, 1027, 1066, 1084, 1143, 1144, 1220, 1221, 1222, 1533, 1776, 2003, 2328, 2402, 2407, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2646, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2690, 2691, 2692, 2706, 2707, 2708, 2709, 2710, 2711, 2857, 2874, 3004, 3005, 3040, 3049, 3080, 3091, 3098, 3253, 3254, 3255, 3256, 3257, 3258, 3266, 3286, 3291, 3319, 3369, 3379, 3422, 3535, 3582, 3589, 3604, 3797, 3798, 3799, 3801, 3807, 3908, 3912, 3990, 3992, 4127, 4128, 4129, 4213, 4219, 4251, 4265, 4309, 4382, 4383, 4528, 4571, 4631, 4654, 4657, 5139, 5270, 5313, 5315, 5600, 5601, 5686, 5715, 5716, 5740, 5742, 5792, 5850]
chunk_over_5_loss.sort(reverse=True)

for item in chunk_over_5_loss:
    del pre_train_tensors_list[item]
print(len(pre_train_tensors_list))
#to do after the first run to find out the errors
#[pre_train_tensors_list[i] for i in range(len(pre_train_tensors_list)) if i not in chunk_over_5_loss]
#
random.seed(1)
random.shuffle(pre_train_tensors_list)

for epoch in range(5):
    errors = []            
    for tensor_200hz in pre_train_tensors_list:
        tensor_200hz = tensor_200hz#.to(device)
        masked_segments, binary_mask, original_segments = partition_and_mask(tensor_200hz,segment_size=10,mask_percentage=0.8)
        masked_segments = masked_segments.to(torch.float32)#.to(device)#.float()
        original_segments = original_segments.to(torch.float32)#.to(device)#.float()
        
        # Encode only the unmasked segments
        encoded_segments = pretrained_encoder(masked_segments,binary_mask)#.to(device)

        # Expand the binary mask to match the embedded dimensions properly for my case we are using 16 which is decoder.num_heads
        binary_mask_expanded = binary_mask.unsqueeze(-1).repeat(1, 1, 1, decoder.num_heads).view(binary_mask.size(0), binary_mask.size(1), -1)#.to(device)
        boolean_mask_expanded = binary_mask_expanded.to(torch.bool)#.to(device) #turn mask into boolean mask 


        reconstructed_data = decoder(encoded_segments, boolean_mask_expanded)#.to(device)
        reconstructed_data = reconstructed_data.to(torch.float32)#.to(device)

        # Calculating reconstruction loss 
        reconstruction_loss = nn.MSELoss()#.to(device) #or mean average error: reconstruction_loss = nn.L1Loss() #MAE are scalable learners uses MSEloss too
        #print(step)
        

        loss = reconstruction_loss(reconstructed_data, original_segments)
        
        # Backpropagate and update weights
        optimizer = optim.Adam(list(pretrained_encoder.parameters()) + list(decoder.parameters()), lr=0.001,weight_decay=1e-5) #MAE are scalable learners uses Adam, weight_decay is regularisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() > 5:
            errored_steps.append(step)
            errors.append(step * 200)
        total_loss += loss.item()
        step+=1
        #errors.append(loss.item())
        
        writer.add_scalar('Loss/Reconstruction_pre', loss.item(), step) 

    writer.add_scalar('Loss/Average_loss_pre', total_loss/step, epoch)
    #with train_summary_writer.as_default():
    #print(f"average loss:{epoch}", total_loss/step)
    #print(f"reconstruction_loss_list:{epoch} ", errors)

#this created the pretrained encoder for later use in the pipeline
pretrained_encoder

#now for comparison, the untrained encoder
untrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.1)


#to-do, look into the high peaks of these cases. The numbers behind length_
#       are the files_names from the data.
#       # I notice that there are some measures where for some acceleration value 
#       there are not 8 decimals. I assume this is a rounding issue. 
#       So when there is a value like 0.32, i change it to 0.32000000 
#       Not sure if that will solve the high errors

print(errored_steps)
#losses higher than 5: 
#chunk_over_5_loss = errored_steps #errored_steps is the list below. This is what filters out the erroring data
chunk_over_5_loss =[46, 167, 250, 299, 301, 302, 340, 360, 392, 393, 394, 491, 492, 493, 611, 615, 695, 699, 757, 814, 821, 844, 871, 907, 908, 909, 910, 911, 912, 913, 918, 946, 988, 999, 1008, 1023, 1026, 1027, 1066, 1084, 1143, 1144, 1220, 1221, 1222, 1533, 1776, 2003, 2328, 2402, 2407, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2646, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2690, 2691, 2692, 2706, 2707, 2708, 2709, 2710, 2711, 2857, 2874, 3004, 3005, 3040, 3049, 3080, 3091, 3098, 3253, 3254, 3255, 3256, 3257, 3258, 3266, 3286, 3291, 3319, 3369, 3379, 3422, 3535, 3582, 3589, 3604, 3797, 3798, 3799, 3801, 3807, 3908, 3912, 3990, 3992, 4127, 4128, 4129, 4213, 4219, 4251, 4265, 4309, 4382, 4383, 4528, 4571, 4631, 4654, 4657, 5139, 5270, 5313, 5315, 5600, 5601, 5686, 5715, 5716, 5740, 5742, 5792, 5850]

#discuss these sequences in the discussion, why they might be wrong..
#pose as a limitation, and drop the troubling sets
#see how the distrubtions differ, pretraining might focus the pipeline more on either more general data or more specified data in comparisiono with the labelled data. 

chunk_over_5_loss = [x * 200 for x in chunk_over_5_loss]

length_640 = 202599
length_642 = 175399 #202 600
length_659 = 400799 #378 000
length_672 = 277599 #778 800
length_676 = 139999 #1 056 400  #1 196 400


chunk_over_5_loss_in_640 = [x for x in chunk_over_5_loss if x <= 202600]
chunk_over_5_loss_in_642 = [(x - 202600) for x in chunk_over_5_loss if 202600 <= x < 378000]
chunk_over_5_loss_in_659 = [(x - 377999) for x in chunk_over_5_loss if 378000 <= x < 778800]
chunk_over_5_loss_in_672 = [(x - 777999) for x in chunk_over_5_loss if 778000 <= x < 1056400]
chunk_over_5_loss_in_676 = [(x - 1056399) for x in chunk_over_5_loss if 1056400 <= x < 1400000]

#this was a check for the csv files. In there not to much seemed wrong. Sometimes there were some decimals missing, 
#one acceleration was hampering a bit, but it was not consistently performing well in other area's too
#The reason I let them in the first time was because I was uncertain whether it was just because of these particular
#samples indicating a new behaviour. But after more consideration the decision was taken to drop the indices causing
#the high reconstruction errors. They will be removed in the section above

