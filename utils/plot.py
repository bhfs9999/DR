import matplotlib.pyplot as plt
import cv2
import numpy as np

color2vec = {
    'g': (0, 255, 0),  # gt
    'r': (255, 0, 0),  # predict
    'b': (0, 0, 255),
}

font = cv2.FONT_HERSHEY_SIMPLEX


def draw_bboxes(img, bboxes, color, thick=4):
    for bbox in bboxes:
        bbox = [int(x) for x in bbox]
        cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4]), color2vec[color], thick)
    return img


def draw_text(img, bboxes, color, texts, thick=3):
    for i, text in enumerate(texts):
        bbox = [int(x) for x in bboxes[i]]
        cv2.putText(img, str(text), tuple(bbox[0:2]), font, 0.5, color2vec[color], thick)

def backtrans_mean(img, mean):
    img += np.array(mean)
    return img

def draw_bboxes_pre_label(img, bboxes_pre, bboxes_label,
                          means, scores=None, classes_pre=None, labels=None, is_ratio=True, CHW=True):
    '''

    :param img: img (C, H, W) or (H, W, C)
    :param bboxes_pre: np list of
    :param bboxes_label:
    :param scores:
    :param labels:
    :param is_ratio:
    :param CHW:
    :return:
    '''
    if CHW:
        img = np.transpose(img, (1, 2, 0)).copy()
    else:
        img = img.copy()
    img = backtrans_mean(img, means).astype(np.uint8)
    bboxes_pre = np.array(bboxes_pre.copy())
    bboxes_label = np.array(bboxes_label.copy())
    if is_ratio:
        ratio_to_real(img, bboxes_pre)
        ratio_to_real(img, bboxes_label)
        # bboxes_pre   = ratio_to_real(img, bboxes_pre, CHW)
        # bboxes_label = ratio_to_real(img, bboxes_label, CHW)
    img = draw_bboxes(img, bboxes_pre, 'r')
    img = draw_bboxes(img, bboxes_label, 'g')

    if scores is not None and classes_pre is not None:
        texts = [str(classes_pre[i]) + ':' + str(scores[i]) for i in range(len(classes_pre))]
        draw_text(img, bboxes_pre, 'r', texts,)
    if labels is not None:
        draw_text(img, bboxes_label, 'g', labels)
    return img


def ratio_to_real(img, bboxes):
    if len(bboxes) == 0:
        return bboxes
    H, W, C = img.shape
    bboxes[:, 0] *= W
    bboxes[:, 1] *= H
    bboxes[:, 2] *= W
    bboxes[:, 3] *= H
