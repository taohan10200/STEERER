import cv2
import numpy as np

Whilte = [255, 255, 255]


def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour)
    return area


def find_max_and_second_large_num(list):
    first = list[0]
    second = 0
    third = 0
    for i in range(1, len(list)):
        if list[i] > first:
            second = first
            first = list[i]
            index_first = i
        elif list[i] > second:
            second = list[i]
            index_second = i

        elif list[i] > third:
            third = list[i]
            index_third = i
    return index_first, index_second, index_third


def get_overlap_region(patch_array, original_size):
    h = original_size[2]
    w = original_size[3]
    overlap_map = np.zeros((h, w))
    # print(patch_array)
    for i in range(len(patch_array)):
        box = patch_array[i]
        overlap_map[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])] += 1
    return overlap_map


def bboverlap(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1
    # print(x0, y0, w0, h0 , x1, y1, w1, h1)
    if x0 > x1 + w1 or x1 > x0 + w0 or y0 > y1 + h1 or y1 > y0 + h0:
        return False

    else:
        return True


def findmaxcontours(distance_map, fname):
    # distance_map = distance_map.astype(np.uint8)
    # print(np.mean(distance_map),np.max(distance_map))
    original_density_map = distance_map

    lambada = 2

    threshold = min(255 * (np.mean(distance_map) * 4 ) / np.max(distance_map),150)
    # if threshold==150:
    #     print ("threshold is 150")

    distance_map = 255 * distance_map / np.max(distance_map)
    distance_map = distance_map[0][0]
    distance_map[distance_map < 0] = 0
    # distance_map[0,0]=255


    img = distance_map.astype(np.uint8)


    ret, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./middle_process/binary2.jpg", binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    list_index = []
    for i in range(len(contours)):
        list_index.append(areaCal(contours[i]))
    list_index.sort(reverse=True)

    first = list_index[0]
    first_index = 0
    img = cv2.applyColorMap(img, 2)

    for i in range(len(contours)):
        if areaCal(contours[i]) == first:
            first_index = i
        # x, y, w, h = cv2.boundingRect(contours[i])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 255), 2)

    cv2.drawContours(img, contours[first_index], -1, (0, 0, 255), 2)
    x, y, w, h = cv2.boundingRect(contours[first_index])
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    coordinate_first = [x, y, w, h]

    save_path = "/mnt/petrelfs/hantao/2023/Autoscale/AutoScale_regression-f0c1583123c16bb726b239c5625834e7b01b17d0/middle_process/contours_result_mean/" + fname[0]
    # import pdb
    # pdb.set_trace()
    save_path = save_path+'.jpg'
    # # ori_distance = cv2.applyColorMap(img,2)
    # # cv2.imwrite("./middle_process/ori_distance.jpg", ori_distance)
    cv2.imwrite(save_path, img)
    return coordinate_first
