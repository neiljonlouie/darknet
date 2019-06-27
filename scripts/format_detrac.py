# Usage: python format_detrac.py /path/to/images /path/to/annotations

import os, shutil, sys
import xml.etree.ElementTree as ET

label_nums = {'bus' : 0, 'car' : 1, 'others' : 2, 'van' : 3}
train_size = 0.8

image_dir = sys.argv[1]
annot_dir = sys.argv[2]

output_dir = 'yolo'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Flatten image directory (uncomment only if needed)
# for subdir in os.listdir(image_dir):
#     for image in os.listdir(os.path.join(image_dir, subdir)):
#         old_image_filename = os.path.join(image_dir, subdir, image)
#         new_image_filename = '%s-%s' % (subdir, image)
#         print('Saving image %s...' % new_image_filename)
#         shutil.copy(old_image_filename, \
#             os.path.join(output_dir, new_image_filename))

# Read annotation file and convert to needed format
dir_list = os.listdir(annot_dir)
num_train_dirs = int(train_size * len(dir_list))

train_file = open('train.list', 'w')
valid_file = open('valid.list', 'w')

img_width = 960
img_height = 540

counter = 0
for file_name in dir_list:
    annot_tree = ET.parse(os.path.join(annot_dir, file_name))
    annot_root = annot_tree.getroot()

    dir_name = annot_root.attrib['name']
    for frame in annot_root.findall('frame'):
        frame_num = int(frame.attrib['num'])

        output_filename = '%s-img%05d.txt' % (dir_name, frame_num)
        output_file = open(os.path.join(output_dir, output_filename), 'w')
        
        print('Writing label file %s...' % output_filename)

        num_objects = int(frame.attrib['density'])
        if num_objects != 0:
            target_list = frame.find('target_list')
            targets = target_list.findall('target')
            for target in targets:
                box = target.find('box')
                attribute = target.find('attribute')
                
                name = attribute.attrib['vehicle_type']
                name_label = label_nums[name]

                xmin = float(box.attrib['left']) / img_width
                ymin = float(box.attrib['top']) / img_height
                width = float(box.attrib['width']) / img_width
                height = float(box.attrib['height']) / img_height

                xcenter = xmin + width / 2
                ycenter = ymin + height / 2

                output_file.write('%d %0.6f %0.6f %0.6f %0.6f\n' % \
                    (name_label, xcenter, ycenter, width, height))  
        
        output_file.close()

        img_filename = '%s-img%05d.jpg' % (dir_name, frame_num)
        if counter >= num_train_dirs:
            valid_file.write('yolo/%s\n' % img_filename)
            # valid_file.write('%s\n' % \
            #     os.path.realpath(os.path.join(output_dir, img_filename)))
        else:
            train_file.write('yolo/%s\n' % img_filename)
            # train_file.write('%s\n' % \
            #     os.path.realpath(os.path.join(output_dir, img_filename)))

    counter += 1

train_file.close()
valid_file.close()

