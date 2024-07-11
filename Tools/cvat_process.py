import av
import glob
import os
import shutil
# import json
import random
# import yaml

import xml.etree.ElementTree as ET

from cvat_sdk import Client, models
from cvat_sdk.core.proxies.tasks import ResourceType
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from ultralytics import YOLO


def create_cvat_file(file_path_in, file_path_out):
    '''file path in is the parent directory. Sub directories should be in image, labels format.
    + parent_dir
        - images
        - labels'''
    os.mkdir(f'{file_path_out}/obj_train_data')
    labels = os.listdir(f'{file_path_in}/labels')
    # move files from one source folder to cvat folder
    for label in labels:
        shutil.copy(f'{file_path_in}/labels/{label}', f'{file_path_out}/obj_train_data/{label}')
    data_file = f'{file_path_out}/obj.data'
    name_file = f'{file_path_out}/obj.names'
    train_file = f'{file_path_out}/train.txt'
    
    for file in [data_file, name_file, train_file]:
        with open(file, 'w') as f:
            if file is data_file:
                data_f_text = [f'classes = 80', 
                                'train = data/train.txt', 
                                'names = data/obj.names', 
                                'backup = backup/'
                                ]
                f.write('\n'.join(data_f_text))
            if file is name_file:
                # could create a list in a text file of all classes and their mapping?
                pass
            if file is train_file:
                images = os.listdir(f'{file_path_in}/images')
                new_list = []
                for image in images:
                    new_list.append(f'data/obj_train/{image}')
                f.write('\n'.join(sorted(new_list)))
                
    shutil.make_archive('./', 'zip', root_dir='cvat_test/')
    
# Convert YOLO labels to CVAT 1.1 annotation xml

def get_image_size(image_path):
    with av.open(image_path) as container:
        video_stream = container.streams.video[0]
        return video_stream.width, video_stream.height

def get_classes_from_labels(input_dir):
    obj_cls_lst = set()
    for file in Path(input_dir).rglob('*.txt'):
        with open(file, 'r') as f:
            for line in f:
                cls = line.strip().split()[0]
                obj_cls_lst.add(cls)
    #### MAY NOT BE NECESSARY DUE TO HOW CVAT IMPORTS LABELS
    int_lst = [int(x) for x in obj_cls_lst]
    cls_lst = [str(x) for x in range(max(int_lst))]
    return cls_lst
    # return sorted(obj_cls_lst, key=int)

def yolo_to_cvat_again(image_directory, label_directory, weights):
    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    labels = ET.SubElement(task, 'labels')

    model = YOLO(weights)
    names = model.names
    colors = set()
    for i in range(len(names)):
        color = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
        while color not in colors:
            label = ET.SubElement(labels, 'label')
            ET.SubElement(label, 'name').text = f"{names[i]}"
            # ET.SubElement(label, 'color').text ="66FF66"
            ET.SubElement(label, 'color').text = f"{color}"
            ET.SubElement(label, 'type').text = "rectangle"
            ET.SubElement(label, 'attributes').text = " "
            i += 1
            colors.add(color)

    ET.SubElement(meta, 'dumped').text = datetime.now().isoformat()
    lbl_idx = 0
    for filename in os.listdir(f'{label_directory}'):
        if filename.endswith('.txt'):
            yolo_file = Path(label_directory, filename)
            
            with open(yolo_file, 'r') as file:
                lines = file.readlines()

            # Get image size
            img_fn = filename.replace('.txt', '.jpg')
            image_file = str(Path(image_directory, img_fn))
            image_width, image_height = get_image_size(image_file)

            # Convert YOLO image format to CVAT format
            image = ET.SubElement(annotations, 'image', id=str(lbl_idx), name=f'{filename[:-4]}.jpg', width=str(image_width), height=str(image_height))

            for i, line in enumerate(lines):
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                # Convert normalized dimensions to absolute dimensions
                x_center *= image_width
                y_center *= image_height
                width *= image_width
                height *= image_height

                # Convert YOLO format to CVAT format
                xtl = x_center - width / 2
                ytl = y_center - height / 2

                # Convert YOLO box format to CVAT format
                box = ET.SubElement(image, 'box', label=str(names[class_id]), source='manual', occluded='0', xtl=str(xtl), ytl=str(ytl), xbr=str(xtl + width), ybr=str(ytl + height), z_order='0')
            
            lbl_idx += 1

    cvat_file = os.path.join(label_directory, 'annotations.xml')
    ET.ElementTree(annotations).write(cvat_file, encoding='utf-8', xml_declaration=True)

    print(f'CVAT annotations written to {cvat_file}')

# Create task and upload images/videos with labels in cvat 1.1 annotation xml #########################################################################

def extract_labels_from_cvat_xml(xml_file):
    def parse_attribute(attribute):
        return {
            "name": attribute.find("name").text,
            "mutable": attribute.find("mutable").text == 'True',
            "input_type": attribute.find("input_type").text,
            "default_value": attribute.find("default_value").text,
            "values": [value.text for value in attribute.findall(".//values/value")]
        }

    def parse_label(label, id):
        return {
            "name": label.find("name").text,
            "id": id,
            "color": label.find("color").text,
            "type": "rectangle",
            "attributes": [parse_attribute(attr) for attr in label.findall(".//attribute")]
        }

    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels_root = root.find(".//labels")
    labels = [parse_label(label, id) for id, label in enumerate(labels_root.findall(".//label"))]
    return labels

def create_task_and_upload(data_dir, task_name, user, pw, spec, labels=False ):
    cvat_server_url = 'http://localhost:8080'
    client = Client(cvat_server_url)
    client.login((user, pw))
    task_spec = {
        "name": task_name,
        "labels": spec
    }

    task_id = client.tasks.create_from_data(
        spec=task_spec, #JSON for labels
        resource_type=ResourceType.LOCAL,
        resources= glob.glob(data_dir),
        data_params={'image_quality': 75, 'sorting_method': "natural"},
        annotation_path=labels,
        annotation_format='CVAT 1.1'
        )
    
    return task_id

def init_data_process_cvat(input_dir, task_name, label_dir):
    # If data is dir of images, images must be compressed to zip file to be uploaded to cvat.
    load_dotenv()
    username = os.getenv('CVAT_USERNAME')
    pword = os.getenv('CVAT_PASSWORD')
    client = Client('http://localhost:8080')
    client.login((username, pword))

    labels_json = extract_labels_from_cvat_xml(f'{label_dir}/annotations.xml')
    
    create_task_and_upload(
        data_dir = f'{input_dir}.zip', # Have to add var about where input files live
        task_name = task_name, # have to create more meaningful var for task name
        labels = f'{label_dir}/annotations.xml', # have to find a way to convert yolo.txt files to xml in cvat fmt
        user = username,
        pw = pword,
        spec = labels_json,
    )