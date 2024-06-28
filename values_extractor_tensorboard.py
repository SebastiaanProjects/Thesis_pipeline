import os
import tensorflow as tf
import pandas as pd
#from CenterNet_utils import load_model
#from MAE_utils import TimeSeriesMAEEncoder, TimeSeriesMAEDecoder
#import torch.optim as optim
#import torch
#from CenterNet_utils import TimeSeriesCenterNet

#pretrained_encoder = TimeSeriesMAEEncoder(segment_dim=4, embed_dim=64, num_heads=16, num_layers=4, dropout_rate=0.1)
#decoder = TimeSeriesMAEDecoder(embed_dim=4, decoder_embed_dim=64, num_heads=16, num_layers=1, max_seq_length=10, dropout_rate=0.1)#.to(device) #num_mask_tokens=6. 0.1 like in literature
#optimizer = optim.Adam(list(pretrained_encoder.parameters()) + list(decoder.parameters()), lr=0.001,weight_decay=1e-5) #MAE are scalable learners uses Adam, weight_decay is regularisation

#model, optimizer = load_model(pretrained_encoder, optimizer, 0, 0)
#for name, param in model.named_parameters():
#    if param.grad is not None:
#        if torch.isnan(param.grad).any():
#            print(f"NaN detected in gradients of {name} before clipping")
#        print(f"Gradients of {name} - min: {param.min().item()}, max: {param.max().item()}")
#    if param.grad is None:
#        print("this save is bad")



def extract_scalars_from_event_file(event_file_path):
    data = []
    for event in tf.compat.v1.train.summary_iterator(event_file_path):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                data.append([event.step, value.tag, value.simple_value])
    return data

def process_directory(input_dir, output_excel_path):
    all_data = {}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file_path = os.path.join(root, file)
                metric_name = os.path.basename(root)
                print(f"Processing: {metric_name}")
                scalar_data = extract_scalars_from_event_file(event_file_path)
                
                if scalar_data:
                    df = pd.DataFrame(scalar_data, columns=['step', 'tag', 'value'])
                    all_data[metric_name] = df
                else:
                    print(f"No scalar data found for: {metric_name}")

    # Write all data to an Excel file
    if all_data:
        with pd.ExcelWriter(output_excel_path) as writer:
            for metric_name, df in all_data.items():
                sheet_name = metric_name[:31]  # Sheet names are limited to 31 characters
                print(f"Writing sheet: {sheet_name}")
                df.to_excel(writer, sheet_name=sheet_name)

        print(f"Data has been saved to {output_excel_path}")
    else:
        print("No data to write to the Excel file.")

# Input directory containing event files
#input_dir = ### INSERT PATH TO FOLDER FROM A PARTICULAR RUN
# Output Excel file
#output_excel_path = ### INSERT PATH TO STORE THE EXCEL FILE CONTAINING THE VALUES

# Process the directory and save metrics to an Excel file
#process_directory(input_dir, output_excel_path)
