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
    '视网膜前出血':0,
    '视网膜深层出血':1,
}

class DetectionDataset(data.Dataset):
    def __init__(self, img_root, xml_root, args, transform=None,):
        self.img_root = img_root
        self.samples, self.fname2labels = self.get_samples(xml_root)
        self.crop = Crop(args.crop_size, args.shift_rate, args.pad_value)
        self.transform = transform

    def __getitem__(self, index):
        labels = []
        fname, label = self.samples[index]
        fpath = os.path.join(self.img_root, fname.lower())
        image = np.array(pil_loader(fpath))
        width, height, channel = image.shape

        # crop image
        image, coord_crop, new_bbox = self.crop(image, label)
        new_label = list(new_bbox) + list(label[-2:])
        labels.append(new_label)

        # fill labels
        all_label = self.fname2labels[fname]
        filled_labels = self._fill_labels(coord_crop, width, height, all_label, label)
        labels += filled_labels
        labels = np.array(labels)

        # data aug
        if self.transform is not None:
            image, boxes, n_class = self.transform(image, labels[:, :4], labels[:, 4])
            labels = np.hstack((boxes, np.expand_dims(n_class, axis=1)))
        # return torch.from_numpy(image).permute(2, 0, 1), labels
        return image, labels[:, :5]

    def __len__(self):
        return len(self.samples)

    def get_samples(self, xml_root):
        """
        :param xml_root
        :return:
            samples: a list of samples like ((fname1, lable1), (fname1, label2), (fname2, label1)...)
            fname2labels: a dict map fname to its all labels (x1, y1, x2, y2, class, idx)
        """
        xml_filenames = os.listdir(xml_root)
        fname2labels = OrderedDict()
        samples = []
        for xml_filename in xml_filenames:
            xml_filepath = os.path.join(xml_root, xml_filename)
            xml_tree = ET.parse(xml_filepath).getroot()
            fname2labels_one = self._get_fname2labels(xml_tree)
            fname2labels.update(fname2labels_one)
            for fname, labels in fname2labels_one.items():
                for label in labels:
                    samples.append([fname, label])
        return samples, fname2labels

    def _get_fname2labels(self, xml_tree):
        fname2labels = OrderedDict()
        fname  = xml_tree.find('filename').text.lower()
        size = xml_tree.find('size')

        width  = int(size.find('width').text) - 1
        height = int(size.find('height').text) - 1

        labels = []
        for idx, obj in enumerate(xml_tree.iter('object')):
            dname = obj.find('name').text.strip()
            bbox = obj.find('bndbox')

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

    def _fill_labels(self, coord_crop, width, height, labels, this_label):
        filled_labels = []
        for label in labels:
            if this_label[-1] == label[-1]:
                continue
            xmin, ymin, xmax, ymax = label[:4]
            xmin = xmin * width
            xmax = xmax * width
            ymin = ymin * height
            ymax = ymax * height

            if xmin > coord_crop[0] and ymin > coord_crop[1] \
                and xmax < coord_crop[2] and ymax < coord_crop[3]:
                new_bbox = change_coord([xmin, ymin, xmax, ymax], coord_crop)
                new_label = list(new_bbox) + list(label[-2:])
                filled_labels.append(new_label)
        return filled_labels

class Crop(object):
    def __init__(self, crop_size, shift_rate, pad_value=0):
        self.crop_size  = crop_size
        self.pad_value  = pad_value
        self.pad_value  = pad_value
        self.max_shift_rate = shift_rate

    def __call__(self, image, label):
        width, height, channel = image.shape
        xmin, ymin, xmax, ymax = label[:4]
        xmin = xmin * width
        xmax = xmax * width
        ymin = ymin * height
        ymax = ymax * height

        # shift
        shift_rate = (np.random.rand(2) - 0.5) * 2 * self.max_shift_rate
        x_shift    = (self.crop_size - (xmax - xmin)) / 2. * shift_rate[0]
        y_shift    = (self.crop_size - (ymax - ymin)) / 2. * shift_rate[1]

        # crop
        xmin_crop = int(xmin - (self.crop_size - (xmax-xmin)) / 2 + x_shift)
        ymin_crop = int(ymin - (self.crop_size - (ymax-ymin)) / 2 + y_shift)
        xmax_crop = xmin_crop + self.crop_size
        ymax_crop = ymin_crop + self.crop_size
        image = image[max(xmin_crop, 0):min(xmax_crop, width),
                      max(ymin_crop, 0):min(ymax_crop, height), :]

        # pad
        left_pad   = max(0, -xmin_crop)
        right_pad  = max(0, xmax_crop-width)
        top_pad    = max(0, -ymin_crop)
        bottom_pad = max(0, ymax_crop-height)
        pad = [[left_pad, right_pad], [top_pad, bottom_pad], [0, 0]]
        image = np.lib.pad(image, pad, 'constant', constant_values=self.pad_value)

        # change coord
        new_coord = change_coord([xmin, ymin, xmax, ymax], [xmin_crop, ymin_crop, xmax_crop, ymax_crop])
        return image, (xmin_crop, ymin_crop, xmax_crop, ymax_crop), new_coord

def change_coord(old_cood, new_ori):
    xmin, ymin, xmax, ymax = old_cood
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








