import os
import tensorflow as tf
import pandas as pd

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
input_dir = ### INSERT PATH TO FOLDER FROM A PARTICULAR RUN
# Output Excel file
output_excel_path = ### INSERT PATH TO STORE THE EXCEL FILE CONTAINING THE VALUES

# Process the directory and save metrics to an Excel file
process_directory(input_dir, output_excel_path)
