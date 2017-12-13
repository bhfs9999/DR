import matplotlib.pyplot as plt
import cv2

color2vec = {
    'g' : (0, 255, 0),      # gt
    'r' : (255, 0, 0),      # predict
    'b' : (0, 0, 255),
}

font = cv2.CV_FONT_HERSHEY_SIMPLEX

def draw_bboxes(img, bboxes, color, thick=4):
    for bbox in bboxes:
        bbox = [int(x) for x in bbox]
        cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4]), color2vec[color], thick)

def draw_text(img, bboxes, color, texts, thick=4):
    for i, text in enumerate(texts):
        bbox = [int(x) for x in bboxes[i]]
        cv2.putText(img, str(text), tuple(bbox[0:2]), font, 1, color2vec[color], thick)

def draw_bboxes_pre_label(img, bboxes_pre, bboxes_label, scores=None, labels=None, is_ratio=True, CHW=True):
    if is_ratio:
        bboxes_pre   = ratio_to_real(img, bboxes_pre, CHW)
        bboxes_label = ratio_to_real(img, bboxes_label, CHW)
    draw_bboxes(img, bboxes_pre, 'r')
    draw_bboxes(img, bboxes_label, 'g')

    if scores:
        draw_text(img, bboxes_pre, 'r', scores,)
    if labels:
        draw_text(img, bboxes_label, 'g', labels)

def ratio_to_real(img, bboxes, CHW=True):
    if CHW:
        C, H, W = img.shape
    else:
        H, W, C = img.shape

    new_bboxes = []
    for bbox in bboxes:
        new_bbox = [x * W if i % 2 == 0 else x * H for i, x in enumerate(bbox)]
        new_bboxes.append(new_bbox)
    return new_bboxes