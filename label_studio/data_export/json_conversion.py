"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
import logging
import random
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------
#                                     RLE DecodeRLEToMask
# -------------------------------------------------------------------------------------

# THERE MIGHT BE MORE. IF SO, NEED BETTER LOGIC!!
class PossibleCategories(Enum):
    BRUSH_LABELS = 'brushlabels'
    KEYPOINT_LABELS = 'keypointlabels'
    POLYGON_LABELS = 'polygonlabels'


possible_categories = [category.value for category in PossibleCategories]


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


# Decoder to convert RLE to mask
class DecodeRLEToMask(object):
    def __init__(self, rle):
        self.rle = rle

    def access_bit(self, data, num):
        """from bytes array to bits by num position"""
        base = int(num // 8)
        shift = 7 - int(num % 8)
        return (data[base] & (1 << shift)) >> shift

    def bytes2bit(self, data):
        """get bit string from bytes data"""
        return ''.join([str(self.access_bit(data, i)) for i in range(len(data) * 8)])

    def decode_rle(self):
        """from LS RLE to numpy uint8 3d image [width, height, channel]"""
        input = InputStream(self.bytes2bit(self.rle))
        num = input.read(32)
        word_size = input.read(5) + 1
        rle_sizes = [input.read(4) + 1 for _ in range(4)]

        i = 0
        out = np.zeros(num, dtype=np.uint8)
        while i < num:
            x = input.read(1)
            j = i + 1 + input.read(rle_sizes[input.read(2)])
            if x:
                val = input.read(word_size)
                out[i:j] = val
                i = j
            else:
                while i < j:
                    val = input.read(word_size)
                    out[i] = val
                    i += 1

        return out


# For coco format need unique colors
def generate_random_color():
    """Generates a random hex color"""
    return '#{:02x}{:02x}{:02x}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


#  For coco format to flatten the contours
def flatten(item):
    if isinstance(item, list):
        for subitem in item:
            yield from flatten(subitem)
    else:
        yield item


# -------------------------------------------------------------------------------------
#                                     Create JSON Format
# -------------------------------------------------------------------------------------


# Json template
json_dict = {'categories': [], 'images': [], 'annotations': []}


def generate_contour_from_RLE(rle, result_data):
    # Decode RLE
    decoder = DecodeRLEToMask(rle)
    rle_binary = decoder.decode_rle()

    expected_height, expected_width = result_data['original_height'], result_data['original_width']
    expected_channels = rle_binary.size // (expected_height * expected_width)

    expected_channels = int(expected_channels)
    reshaped_image = rle_binary.reshape(
        (expected_height, expected_width, 4)
    )  # RLE give 4 channels of the same mask... Why??

    # Convert contours to COCO format
    contours, _ = cv2.findContours(
        np.array(reshaped_image[:, :, 0]).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    segmentation = []
    bbox_list = []
    area_list = []
    for contour in contours:
        # Flatten list of points
        contour_flat = contour.flatten().tolist()
        # Check if the segmentation is valid (should have more than 4 points for a polygon)
        if len(contour_flat) > 4:
            segmentation.append(contour_flat)

            x, y, w, h = cv2.boundingRect(contour)
            bbox_list.append([x, y, w, h])   # Format: [top-left x, top-left y, width, height]

            # Compute the area of the contour
            area = cv2.contourArea(contour)
            area_list.append(int(area))

    return segmentation, bbox_list, area_list


def generate_contour_from_polygon(result_data, values):

    # Extract and multiply the points
    modified_points = [
        [int(x * result_data['original_width'] / 100), int(y * result_data['original_height'] / 100)]
        for x, y in values['points']
    ]

    # Initialize lists
    segmentation = [list(flatten(modified_points))]

    # Convert points to a NumPy array and reshape for OpenCV
    contour = np.array(modified_points, dtype=np.float32).reshape((-1, 1, 2))

    # Calculate bounding box
    bbox = cv2.boundingRect(contour)
    bbox = [int(coordinate) for coordinate in bbox]

    # Compute the area of the contour
    area = cv2.contourArea(contour)

    return segmentation, bbox, [area]


def RLE_To_JSON(data):

    # Initialize the template for main keys images
    main_keys_template = {
        'filename': None,
        'id': None,
        'drafts': None,
        'predictions': None,
        'data': None,
        'meta': None,
        'created_at': None,
        'updated_at': None,
        'inner_id': None,
        'total_annotations': None,
        'cancelled_annotations': None,
        'total_predictions': None,
        'comment_count': None,
    }

    # Initialize the template for main keys annotations
    image_json_template = {
        'dataset_id': None,
        'category_ids': [],
        'width': None,
        'height': None,
        'annotated': False,
        'annotating': [],
        'num_annotations': 0,
        'metadata': {},
        'deleted': False,
        'milliseconds': 0,
        'events': [],
        'regenerate_thumbnail': False,
    }

    CAT = []
    for ind, i in enumerate(data):
        filename = i['data']['image'].split('/')[-1]

        # Use the .copy() method to ensure you get a new instance of the dictionary
        MainKeys = main_keys_template.copy()
        MainKeys.update({'filename': filename, 'id': ind})

        image_JSON = image_json_template.copy()
        image_JSON.update(
            {
                'id': ind,
                'path': i['data']['image'],
                'file_name': filename,
            }
        )

        # Iterate through data and append values to MainKeys
        for ky in MainKeys.keys():
            if ky in i and ky != 'annotations' and ky != 'id':
                MainKeys[ky] = i[ky]

        # create blank template of annotation key
        annotation_keys = [
            'id',
            'completed_by',
            'was_cancelled',
            'ground_truth',
            'created_at',
            'updated_at',
            'draft_created_at',
            'lead_time',
            'prediction',
            'result_count',
            'unique_id',
            'import_id',
            'last_action',
            'task',
            'project',
            'updated_by',
            'parent_prediction',
            'parent_annotation',
            'last_created_by',
        ]

        annotation_data = {key: None for key in annotation_keys}

        for anns in i['annotations']:

            for key in annotation_data.keys():
                if key in anns.keys():
                    annotation_data[key] = anns[key]

            annotation_data['id'] = ind

            for annotation in i['annotations']:
                results = annotation['result']

                for ind_ann, result in enumerate(results):

                    annotations_JSON = {
                        'id': None,
                        'image_id': None,
                        'category_id': None,
                        'segmentation': [],
                        'area': None,
                        'bbox': None,
                        'iscrowd': 'false',
                        'isbbox': 'false',
                        'color': '#d446da',
                        'metadata': {},
                    }

                    result_keys = [
                        'original_width',
                        'original_height',
                        'image_rotation',
                        'id',
                        'from_name',
                        'to_name',
                        'type',
                        'origin',
                        'segmentation',
                    ]
                    result_data = {key: None for key in result_keys}

                    for key in result_data.keys():
                        if key in result:
                            result_data[key] = result[key]

                    image_JSON['height'] = result_data['original_height']
                    image_JSON['width'] = result_data['original_width']
                    values = result['value']

                    rle = values.get('rle', None)

                    # if not RLE, skip for now
                    if rle is None and 'polygonlabels' not in values.keys():
                        continue

                    if 'polygonlabels' in values.keys():
                        # If polygon result exists
                        segmentation, bbox, area_list = generate_contour_from_polygon(result_data, values)
                        bbox_list = [bbox]

                    else:
                        # If SAM results exist
                        segmentation, bbox_list, area_list = generate_contour_from_RLE(rle, result_data)

                    annotations_JSON['id'] = ind * 1000 + ind_ann
                    annotations_JSON['image_id'] = ind
                    annotations_JSON['segmentation'] = segmentation
                    annotations_JSON['bbox'] = bbox_list[0]
                    annotations_JSON['area'] = area_list[0]
                    image_JSON['num_annotations'] = ind_ann

                    # Determine the 'categories' value
                    categories = None
                    category_index = []
                    for cat in possible_categories:
                        if cat in values.keys():
                            categories = values[cat]

                            if categories in CAT:
                                category_index.append(CAT.index(categories))
                            else:
                                CAT.append(categories)
                                category_index.append(CAT.index(categories))

                            break

                    annotations_JSON['category_id'] = category_index[0] + 1
                    json_dict['annotations'].append(annotations_JSON)

            json_dict['images'].append(image_JSON)

    # Fill in the categories field base on the number of unique categories
    def add_categories(json_dict, category_names):
        """Adds categories to the given json_dict based on a list of category names"""
        for idx, category_name in enumerate(category_names, start=1):
            category = {
                'id': idx,
                'name': category_name,
                'supercategory': '',
                'color': generate_random_color(),
                'metadata': {},
                'keypoint_colors': [],
            }
            json_dict['categories'].append(category)

    add_categories(json_dict, CAT)

    return json_dict
