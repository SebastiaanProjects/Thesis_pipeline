import os
import tensorflow as tf

def save_image_raw(event_file_path, output_dir):
    for event in tf.compat.v1.train.summary_iterator(event_file_path):
        for value in event.summary.value:
            if value.HasField('image'):
                print(f"Found image tag: {value.tag}")
                try:
                    img_tensor = tf.io.decode_raw(value.image.encoded_image_string, tf.uint8)
                    print(f"Image tensor shape (before reshape): {img_tensor.shape}")
                    
                    # Attempt to decode the image directly
                    img_tensor = tf.image.decode_image(value.image.encoded_image_string)
                    img_file_path = os.path.join(output_dir, f'{value.tag}_{event.step}.png')
                    tf.io.write_file(img_file_path, tf.image.encode_png(img_tensor))
                    print(f"Saved image: {img_file_path}")

                except Exception as e:
                    print(f"Error processing image tag {value.tag}: {e}")

def extract_images_from_event_file(event_file_path, output_dir):
    print(f"Processing file: {event_file_path}")
    save_image_raw(event_file_path, output_dir)

# Path to the specific event file
event_file_path = ### INSERT PATH TO RUN EVENTS.OUT.TFEVENTS FILE FOR A PARTICULAR RUN'
output_dir = ### INSERT PATH TO STORE THE IMAGES
os.makedirs(output_dir, exist_ok=True)

# Extract images from the event file
extract_images_from_event_file(event_file_path, output_dir)

print(f"Images have been saved to {output_dir}")

