from MAE_utils import TimeSeriesMAEDecoder, TimeSeriesMAEEncoder
from data_composer import pre_train_tensors_list
import torch
import torch.nn as nn
import torch.optim as optim
from Data_extraction_utils import partition_and_mask
from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter()

#                                   4               64          16              3           0.1
pretrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.2) #64 may be higher. Just for decoder to decide. 
# use a parameter optimizer to decide, snellius  
decoder = TimeSeriesMAEDecoder(embed_dim=4, decoder_embed_dim=64, num_heads=16, num_layers=1, max_seq_length=10, dropout_rate=0.2) #num_mask_tokens=6
total_loss = 0
step = 0

for epoch in range(20):
    errors = []            
    for tensor_200hz in pre_train_tensors_list:
        masked_segments, binary_mask, original_segments = partition_and_mask(tensor_200hz,segment_size=10,mask_percentage=0.7)
        masked_segments = masked_segments.to(torch.float32)#.float()
        original_segments = original_segments.to(torch.float32)#.float()
        
        # Encode only the unmasked segments
        encoded_segments = pretrained_encoder(masked_segments,binary_mask)

        # Expand the binary mask to match the embedded dimensions properly for my case we are using 16 which is decoder.num_heads
        binary_mask_expanded = binary_mask.unsqueeze(-1).repeat(1, 1, 1, decoder.num_heads).view(binary_mask.size(0), binary_mask.size(1), -1)
        boolean_mask_expanded = binary_mask_expanded.to(torch.bool) #turn mask into boolean mask 


        reconstructed_data = decoder(encoded_segments, boolean_mask_expanded)
        reconstructed_data = reconstructed_data.to(torch.float32)

        # Calculating reconstruction loss 
        reconstruction_loss = nn.MSELoss() #or mean average error: reconstruction_loss = nn.L1Loss() #MAE are scalable learners uses MSEloss too
        #print(step)
        

        loss = reconstruction_loss(reconstructed_data, original_segments)
        
        # Backpropagate and update weights
        optimizer = optim.Adam(list(pretrained_encoder.parameters()) + list(decoder.parameters()), lr=0.001,weight_decay=1e-5) #MAE are scalable learners uses Adam, weight_decay is regularisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step+=1
        if loss.item() > 5:
            print(step)
            errors.append(step * 200)
        total_loss += loss.item()
        #errors.append(loss.item())
        
        writer.add_scalar('Loss/Reconstruction_pre', loss.item(), step) 
        #with train_summary_writer.as_default():
    print("errors",errors)
    print(f"average loss:{epoch}", total_loss/step)
    #print(f"reconstruction_loss_list:{epoch} ", errors)

#this created the pretrained encoder for later use in the pipeline
pretrained_encoder

#now for comparison, the untrained encoder
untrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.2)


#to-do, look into the high peaks of these cases. The numbers behind length_
#       are the files_names from the data.
#       # I notice that there are some measures where for some acceleration value 
#       there are not 8 decimals. I assume this is a rounding issue. 
#       So when there is a value like 0.32, i change it to 0.32000000 
#       Not sure if that will solve the high errors

#losses higher than 5: 
chunk_over_5_loss =[47,168,251,300,302,303,341,361,393,394,395,492,493,494,612,616,696,700,758,815,822,845,872,908,909,910,911,912,913,914,919,947,989,1000,1009,1024,1027,1028,1067,1085,1144,1145,1221,1222,1223,1534,1777,2004,2329,2403,2408,2471,2472,2473,2474,2475,2476,2477,2478,2479,2480,2647,2650,2651,2652,2653,2654,2655,2656,2657,2658,2659,2660,2661,2662,2663,2664,2665,2666,2667,2668,2669,2670,2671,2672,2673,2674,2675,2676,2677,
2678,679,2691,2692,2693,2707,2708,2709,2710,2711,2712,2858,2875,3005,3006,3041,3050,3081,3092,3099,3254,3255,3256,3257,
3258,3259,3267,3287,3292,3320,3370,3380,3423,3536,3583,3590,3605,3798,3799,3800,3802,3808,3909,3913,3991,3993,
4128,4129,4130,4214,4220,4252,4266,4310,383,4384,4529,4572,4632,4655,4658,5140,5271,5314,5316,5601,5602,5687,5716,5717,5741,5743,5793,5851
]

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

