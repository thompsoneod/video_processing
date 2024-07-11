import argparse
import os
import Tools.vid_process as vp
import Tools.model as mdl
import Tools.cvat_process as cv
import shutil

# from cvat_sdk import Client
from pathlib import PurePath
from pathlib import Path


ROOT = os.getcwd()

# Image Processing
def key_frame(args):
    vp.extract_key_frames(args.input_dir, args.output_dir)

def downsample(args):
    vp.downsample_video(args.input_dir, args.output_dir, args.down_sample)

# Inference
def inference(args):
    if args.key_frame:
        print('Running inference on key frames...')
        input_dir = f'{args.output_dir}{args.name}/key_frames/'
        mdl.inf_model(input_dir, args.weights, args.name)
    elif args.down_sample:
        print('Running inference on down sampled frames...')
        input_dir = f'{args.output_dir}{args.name}/down_sample/'
        mdl.inf_model(input_dir, args.weights, args.name)
    else:
        print(f'Running inference on {args.output_dir}{args.name}/key_frames/')
        input_dir = f'{args.output_dir}{args.name}/key_frames/'
        mdl.inf_model(input_dir, args.weights, args.name)

# Upload images to CVAT
def cvat(args):
    labels_dir = f'{ROOT}/predictions/{args.name}/labels/'
    output_dir = f'{args.output_dir}{args.name}/key_frames'
    
    if args.key_frame:
        output_dir = f'{args.output_dir}{args.name}/key_frames'
    elif args.down_sample:
        output_dir = f'{args.output_dir}{args.name}/down_sample'
        
    print('Creating CVAT annotations...')
    cv.yolo_to_cvat_again(output_dir, labels_dir, args.weights)
    print('Zipping key frames...')
    shutil.make_archive(f'{args.output_dir}{args.name}', 'zip', root_dir=f'{output_dir}')
    print('Uploading to CVAT...')
    cv.init_data_process_cvat(f'{args.output_dir}{args.name}', args.name, labels_dir)
    print('Upload complete!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=Path(f'{ROOT}/media'), help="Directory where source images are located.")
    parser.add_argument('--output_dir', type=str, default = f"{ROOT}/images/", help="Directory where source labels are to be saved.")
    parser.add_argument('--name', type=str, help="Name of task in CVAT.")
    parser.add_argument('--down_sample', '-ds', type=int, help="How many frames do you want per second.")
    parser.add_argument('--weights', type=str, default = Path(f"{ROOT}/weights"), help=".pt file for custom trained model.")
    # parser.add_argument('--weights', type=str, default = f"{ROOT}/torun_v3.pt.pt", help=".pt file for custom trained model.")
    parser.add_argument('--cvat', '-cvat', action='store_true', help="set true if attempting to retrain model on data")
    parser.add_argument('--predict', '-p', action='store_true', help="set true if attempting to run predictions on model data")
    parser.add_argument('--key_frame', '-kf', action='store_true', help="Set false if not attempting to extract key frames.")
    args = parser.parse_args()
    

    
    # file_name = PurePath(args.input_dir).stem
    actions = {
        'key_frame': key_frame,
        'down_sample': downsample,
        'predict': inference,
        'cvat': cvat,
    }
    
    # If Downsample is specified then key sample isn't run.
    if args.down_sample:
        actions.pop('key_frame')

    # Select file if multiple files in media source directory
    if len(list(args.input_dir.rglob('*.*'))) == 1:
        input_dir = list(args.input_dir.rglob('*.*'))
        args.input_dir = str(input_dir[0])
        print(args.input_dir)

    else:
        print('Too many files in directory. Please specify one file.')
        input_dir = list(args.input_dir.rglob('*.*'))
        for i, item in enumerate(input_dir):
            print(f'{i}: {item}') 
        src = int(input('Enter the number of the file you want to process: '))
        print(f'Processing {input_dir[src]}')
        args.input_dir = str(input_dir[src])
    
    # Select weights file if multiple files in weights directory
    if args.predict:
        if len(list(args.weights.rglob('*.pt'))) == 1:
            input_wts = list(args.input_dir.rglob('*.pt'))
            args.weights = str(input_wts[0])
            print(args.weights)

        else:
            print('Which weights file would you like to use. Please specify one file.')
            input_wts = list(args.weights.rglob('*.*'))
            for i, item in enumerate(input_wts):
                print(f'{i}: {item}') 
            src = int(input('Enter the number of the file you want to process: '))
            print(f'Processing {input_wts[src]}')
            args.weights = str(input_wts[src])
    else:
        args.weights = args.weights/'yolov8l.pt'
    
    
    if not args.name:
        args.name = PurePath(args.input_dir).stem
    
    if Path(args.input_dir).suffix == '.zip':
        os.makedirs(f'{ROOT}/images/{args.name}/key_frames/')
        shutil.unpack_archive(args.input_dir, f'{ROOT}/images/{args.name}/key_frames/')
    else:
        vp.vid_metadata(args.input_dir)
    
    for arg, action in actions.items():
        if getattr(args, arg, None):
            action(args)
            
if __name__=='__main__':
    main()