# Usage: python3 tracker_py /path/to/json/file

import json, os, sys
import cv2

iou_threshold = 0.5
image_height = 540
image_width = 960
colors = [(255,0,0), (0,255,0), (255,255,0), (0,0,255)]

def compute_iou(track, detection):
    rel_coords = detection['relative_coordinates']
    center_x = rel_coords['center_x'] * image_width
    center_y = rel_coords['center_y'] * image_height
    width = rel_coords['width'] * image_width
    height = rel_coords['height'] * image_height

    x1 = max(track['xmin'], center_x - 0.5 * width)
    y1 = max(track['ymin'], center_y - 0.5 * height)
    x2 = min(track['xmin'] + track['width'], center_x + 0.5 * width)
    y2 = min(track['ymin'] + track['height'], center_y + 0.5 * height)

    if x1 > x2 or y1 > y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = track['width'] * track['height']
    area2 = width * height
    union = area1 + area2 - intersection
    return intersection / union


json_filename = sys.argv[1]
json_file = open(json_filename)

frames = json.load(json_file)
frames = sorted(frames, key=lambda x: x['frame_id'])

output_dir = 'results'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tracks = []
curr_id = 0
for frame in frames:
    detections = frame.get('objects', [])
    for track in tracks:
        # Find best matching detection per track
        updated = False
        if len(detections) > 0:
            best_match = max(detections, key=lambda x: compute_iou(track, x))
            if compute_iou(track, best_match) >= iou_threshold:
                rel_coords = best_match['relative_coordinates']
                center_x = rel_coords['center_x'] * image_width
                center_y = rel_coords['center_y'] * image_height
                width = rel_coords['width'] * image_width
                height = rel_coords['height'] * image_height

                track['xmin'] = center_x - 0.5 * width
                track['ymin'] = center_y - 0.5 * height
                track['height'] = height
                track['width'] = width
                track['class_id'] = best_match['class_id']
                track['name'] = best_match['name']
                del detections[detections.index(best_match)]
                updated = True

        if not updated:
            track['track_id'] = -1
    
    tracks = [track for track in tracks if track['track_id'] > 0]

    # Add unmatched detections as new tracks
    new_tracks = []
    for detection in detections:
        track = {}
        curr_id += 1
        track['track_id'] = curr_id
        track['class_id'] = detection['class_id']
        track['name'] = detection['name']

        rel_coords = detection['relative_coordinates']
        center_x = rel_coords['center_x'] * image_width
        center_y = rel_coords['center_y'] * image_height
        width = rel_coords['width'] * image_width
        height = rel_coords['height'] * image_height

        track['xmin'] = center_x - 0.5 * width
        track['ymin'] = center_y - 0.5 * height
        track['height'] = height
        track['width'] = width

        new_tracks.append(track)

    tracks = tracks + new_tracks
    
    image_filename = frame['filename']
    image = cv2.imread(image_filename)
    for track in tracks:
        xmin = int(track['xmin'])
        ymin = int(track['ymin'])
        width = int(track['width'])
        height = int(track['height'])

        cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height),
                      colors[track['class_id']], 1)
        cv2.putText(image, '#%d: %s' % (track['track_id'], track['name']), 
                    (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 2.0,
                    colors[track['class_id']], 2)
        
    filename_parts = image_filename.split('/')
    output_filename = os.path.join(output_dir, filename_parts[1])
    print('Saving %s...' % output_filename)
    cv2.imwrite(output_filename, image)

json_file.close()