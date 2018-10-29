import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import sys
from collections import OrderedDict
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

dname2label = {
    '微血管瘤':0, #MA
    '出血斑':1, #HE
}

class DetectionDataset(data.Dataset):
    def __init__(self, img_root, xml_root, filenames, crop_size, shift_rate, pad_value, transform=None, is_test=False):
        self.img_root = img_root
        self.samples, self.fname2labels = self.get_samples(xml_root, filenames)
        self.crop = Crop(crop_size, shift_rate, pad_value)
        self.transform = transform
        self.is_test   = is_test

    def __getitem__(self, index):
        labels = []
        fname, label, id = self.samples[index]
        fpath = os.path.join(self.img_root, fname.lower())
        image = np.array(pil_loader(fpath))

        # crop image
        image, coord_crop, new_bbox = self.crop(image, label)
        new_bbox = np.array(new_bbox)
        np.clip(new_bbox, 0., 1., out=new_bbox)
        new_label = list(new_bbox) + list(label[-2:])
        labels.append(new_label)

        # fill labels
        all_label = self.fname2labels[fname]
        filled_labels = self._fill_labels(coord_crop, all_label, label)
        labels += filled_labels
        labels = np.array(labels)

        # data aug
        if self.transform is not None:
            image, boxes, n_class = self.transform(image, labels[:, :4], labels[:, 4])
            labels = np.hstack((boxes, np.expand_dims(n_class, axis=1)))
        if self.is_test:
            return torch.from_numpy(image).permute(2, 0, 1), labels[:, :5], id
        else:
            return torch.from_numpy(image).permute(2, 0, 1), labels[:, :5],
        # return image, labels[:, :5]

    def __len__(self):
        return len(self.samples)

    def get_samples(self, xml_root, xml_filenames):
        """
        :param xml_root
        :return:
            samples: a list of samples like ((fname1, lable1), (fname1, label2), (fname2, label1)...) (182006, (x1 y1 x2 y2 0, 0))
            fname2labels: a dict map fname to its all labels (x1, y1, x2, y2, class, idx)
        """
        fname2labels = OrderedDict()
        samples = []
        id = 0
        for xml_filename in xml_filenames:
            xml_filepath = os.path.join(xml_root, xml_filename)
            xml_tree = ET.parse(xml_filepath).getroot()
            fname2labels_one = self._get_fname2labels(xml_tree)
            fname2labels.update(fname2labels_one)
            for fname, labels in fname2labels_one.items():
                for label in labels:
                    samples.append([fname, label, id])
                    id += 1
        return samples, fname2labels

    def _get_fname2labels(self, xml_tree):
        fname2labels = OrderedDict()
        fname  = xml_tree.find('filename').text.lower().split('_')[0] + '.jpg'
        size = xml_tree.find('size')

        width  = int(size.find('width').text) - 1
        height = int(size.find('height').text) - 1

        labels = []
        for idx, obj in enumerate(xml_tree.iter('object')):
            dname = obj.find('name').text.strip()
            bbox = obj.find('bndbox')
            if dname not in dname2label.keys():
                continue

            label = []
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                label.append(cur_pt)
            label.append(dname2label[dname])
            label.append(idx)
            labels.append(np.array(label))
        fname2labels[fname] = labels
        return fname2labels

    def _fill_labels(self, coord_crop, labels, this_label):
        filled_labels = []
        for label in labels:
            if this_label[-1] == label[-1]:
                continue
            xmin, ymin, xmax, ymax = label[:4]
            x_center = (xmax + xmin) / 2
            y_center = (ymax + ymin) / 2

            if x_center > coord_crop[0] and y_center > coord_crop[1] \
                and x_center < coord_crop[2] and y_center < coord_crop[3]:
                new_bbox = change_coord([xmin, ymin, xmax, ymax], coord_crop)

                # clip
                new_bbox = np.array(new_bbox)
                np.clip(new_bbox, 0., 1., new_bbox)
                new_label = list(new_bbox) + list(label[-2:])
                filled_labels.append(new_label)
        return filled_labels

class Crop(object):
    def __init__(self, crop_size, shift_rate, pad_value=0):
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.pad_value = pad_value
        self.max_shift_rate = shift_rate

    def __call__(self, image, label):
        '''
        crop image and get new coord
        :param image: images
        :param label: label (x1, y1, x2, y2, cls)
        :return:
            image     : crop image (type: np.array)
            coord_crop: list of cropped image's coord(ratio) to ori image
            new_coord : new coord(ratio) to cropped image
        '''
        height, width, channel = image.shape        # 1017 1160
        xmin, ymin, xmax, ymax = label[:4]
        xmin_t = xmin * width
        xmax_t = xmax * width
        ymin_t = ymin * height
        ymax_t = ymax * height
        # print(xmin_t, ymin_t, xmax_t, ymax_t)

        # shift
        shift_rate = (np.random.rand(2) - 0.5) * 2 * self.max_shift_rate
        x_shift_t  = (self.crop_size - (xmax_t - xmin_t)) / 2. * shift_rate[0]
        y_shift_t  = (self.crop_size - (ymax_t - ymin_t)) / 2. * shift_rate[1]

        # crop
        xmin_crop_t = int(xmin_t - (self.crop_size - (xmax_t-xmin_t)) / 2 + x_shift_t)
        ymin_crop_t = int(ymin_t - (self.crop_size - (ymax_t-ymin_t)) / 2 + y_shift_t)
        xmax_crop_t = xmin_crop_t + self.crop_size
        ymax_crop_t = ymin_crop_t + self.crop_size
        image = image[max(ymin_crop_t, 0):min(ymax_crop_t, height),
                      max(xmin_crop_t, 0):min(xmax_crop_t, width), :]

        # crop true value to ratio
        xmin_crop  = float(xmin_crop_t) / width
        ymin_crop  = float(ymin_crop_t) / height
        xmax_crop  = float(xmax_crop_t) / width
        ymax_crop  = float(ymax_crop_t) / height
        coord_crop = [xmin_crop, ymin_crop, xmax_crop, ymax_crop]

        # pad
        left_pad   = max(0, -xmin_crop_t)
        right_pad  = max(0, xmax_crop_t-width)
        top_pad    = max(0, -ymin_crop_t)
        bottom_pad = max(0, ymax_crop_t-height)
        pad = [[top_pad, bottom_pad],[left_pad, right_pad], [0, 0]]
        image = np.lib.pad(image, pad, 'constant', constant_values=self.pad_value)

        # change coord
        new_coord = change_coord([xmin, ymin, xmax, ymax], coord_crop)
        return image, coord_crop, new_coord

def change_coord(old_coord, new_ori):
    '''
    change old coord to new_ori
    :param old_cood: old coord(ratio)
    :param new_ori:  new coord(ratio)
    :return:
        a list of new coord(ratio) to new ori
    '''
    xmin, ymin, xmax, ymax = old_coord
    x_ori, y_ori, x_end, y_end = new_ori

    width  = x_end - x_ori
    height = y_end - y_ori

    x_min_new = (xmin - x_ori) / width
    x_max_new = (xmax - x_ori) / width
    y_min_new = (ymin - y_ori) / height
    y_max_new = (ymax - y_ori) / height
    return x_min_new, y_min_new, x_max_new, y_max_new

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
