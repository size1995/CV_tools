import numpy as np
import math
def put_heatmap(heatmap, center, sigma):
    center_x, center_y = center  #
    height, width = heatmap.shape[:2]
    th = 4.6052
    delta = math.sqrt(th * 2)
    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))
    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))
    exp_factor = 1 / 2.0 / sigma / sigma
    arr_heatmap = heatmap[ y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    arr_exp = arr_exp/np.max(arr_exp)
    heatmap[ y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap

def draw_heatmap(class_N,height,width,class_ids,points,sigma):
    """
    :param class_N: int number of class
    :param height:  int height of heatmap
    :param width: int width of heatmap
    :param class_ids: list class ids for all the points [a,b]
    :param points: list of list [[[x1,y1],[x2,y2]],[[x3,y3],[x4,y4]]]
    :param heatmap: numpy array
    :return: heatmap
    """
    heatmap = np.zeros(shape=(class_N,height,width), dtype=np.float32)
    if len(points)!=len(class_ids):
        raise Exception("points and class_ids do not match in length!")
    for i in range(len(class_ids)):
        class_i = class_ids[i]
        point_list = points[i]
        for point in point_list:
            heatmap[class_i] = put_heatmap(heatmap[class_i], [int(width*point[0]),int(height*point[1])], sigma)

