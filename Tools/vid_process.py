import av
import json
import cv2
import os
from tqdm import tqdm
import shutil


# EXTRACT METADATA
def vid_metadata(video_path):
    container = av.open(video_path)

    metadata = {
        'Codec_name' : container.streams.video[0].codec.long_name,
        'Color Channels' : container.streams.video[0].format.name,
        'Duration (Seconds)': container.duration/av.time_base,
        'Height':  container.streams.video[0].height,
        'Width' : container.streams.video[0].width,
        'file_size_bytes' : container.size,
        'Metadata': container.metadata
    }

    file_name = video_path.split('/')[-1]
    creation_time = metadata['Metadata']['creation_time']
    try:
        make_model = f"{metadata['Metadata']['com.android.manufacturer']}_{metadata['Metadata']['com.android.model']}"
    except:
        make_model = f"{metadata['Metadata']['artist']}"

    json_file_Name = f"{file_name}_{make_model}_TIME_{creation_time}.json"
    with open(json_file_Name, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
        
############### VIDEO PROCESSING ###############################################################################################
# SPLIT KEY FRAMES
def extract_key_frames(file_path, save_path):
    '''
    INPUTS:
    file_path: path of file to be be extracted
    save_path: path to save file to
    name: names to save images as
    '''
    name = os.path.splitext(os.path.basename(file_path))[0]
    save_fldr = os.path.join(save_path, name, 'key_frames')
    os.makedirs(save_fldr)
    
    def count_key_frames(file_path):
        count = 0
        with av.open(file_path) as container:
            # Access the video stream
            stream = container.streams.video[0]
            # Set codec context to skip non-key frames
            stream.codec_context.skip_frame = "NONKEY"
            # Iterate over frames and count key frames
            for _ in container.decode(stream):
                count += 1
        return count
    
    key_frame_count = count_key_frames(file_path)
    
    i=0
    with av.open(file_path) as container:
        # Signal that we only want to look at keyframes.
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        # print(len(stream))
        for frame_index, frame in enumerate(tqdm(container.decode(stream), total=key_frame_count, desc="Extracting key frames")):
            # We use `frame.pts` as `frame.index` won't make must sense with the `skip_frame`.
            # frame.to_image().save(os.path.join(save_fldr, f"{name}_{i}.jpg").format(frame.pts))
            frame.to_image().save(os.path.join(save_fldr, f"{name}_{i}.jpg"))
            i+=1
    print(f"Key frames saved to {save_fldr}")
    
    return save_fldr
    # works but makes more sense to have in cvat process
    # shutil.make_archive(save_fldr, 'zip', save_fldr)
    # shutil.move(f"{save_fldr}.zip", save_path)

# DOWNSAMPLE VIDEO            
def downsample_video(input_video, output_folder, target_frame_rate):
    '''
    INPUTS:
    input_video = "path/to/your/input_video.mp4"
    output_folder = "path/to/your/output_frames"
    target_frame_rate = 10  # Adjust this to your desired frame rate
    '''

    # Open the video file
    cap = cv2.VideoCapture(input_video)

    # Get the original frame rate of the video
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame skip factor to achieve the target frame rate
    frame_skip_factor = int(original_frame_rate / target_frame_rate)

    # Create the output folder if it doesn't exist
    vid_name = os.path.splitext(os.path.basename(input_video))[0]
    output_folder = os.path.join(output_folder, vid_name, 'down_sample')
    os.makedirs(output_folder, exist_ok=True)
    # Read and save frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        # print(frame_count, frame_skip_factor)
        if not ret:
            break  # Break the loop if no more frames are available
        if frame_skip_factor >= 1:
            if frame_count % frame_skip_factor == 0:
                # Save the frame as a .jpg image
                frame_filename = os.path.join(output_folder, f"{vid_name}_frame_{frame_count // frame_skip_factor}.jpg")
                cv2.imwrite(frame_filename, frame)
        else:
            continue

        frame_count += 1
    # Release the video capture object
    cap.release()
    print(f"Frames saved to {output_folder}")