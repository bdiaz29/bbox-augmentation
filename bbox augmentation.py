import xmltodict
import os
import glob
import xmltodict
import xlwt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
from random import randint
import PIL
from PIL import Image
import cv2
import math
import pandas as pd
from tkinter import *

datagen = ImageDataGenerator()

augmentsnum = 25
rowcount = 1
name="default"

window = Tk()
window.title("image bounding box augmentor")
window.geometry('500x500')
model=[]

file_frame = Frame(master=window)
file_frame.grid(column=0, row=0)
source=""
destination=""
destination_set=False
source_set=False
from tkinter import filedialog


# datagen = ImageDataGenerator()
F = 3


def translate(x,y,height,width):
    x=x+(width/2)
    y=y+(height/2)
    y=-1*(y-height)
    return x,y

def max_exist(x, y, side):
    T = translate(x, y, 240, 320)
    A = x ** 2 + y ** 2
    #B = math.sqrt(A)
    C = side**2
    if A > C:
        return True
    else:
        return False


def angle_find_vectors(xa, ya, xb, yb):
    T = translate(xa, yb, 240, 320)
    A = [xa, ya]
    B = [xb, yb]
    # C = [abs(xa), abs(ya)]
    # D = [abs(xb), abs(yb)]
    C = math.sqrt((xa ** 2) + (ya ** 2))
    D = math.sqrt((xb ** 2) + (yb ** 2))

    F = np.dot(A, B)
    G = C * D
    #if for whateve reason it is one (outside the domain of acos)
    #return zero
    if abs(F/G)>=1:
        return 0

    H = math.acos(F / G)
    return H


def angle_find2(x, y, side, hs, ws, height_width):
    T=translate(x,y,240,320)
    exist = False
    angle = 1.5708
    width=1
    height=1
    a=height_width
    exist = max_exist(x, y, side)
    if not exist:
        return angle

    one_side = x ** 2 + y ** 2
    side_sq = side ** 2
    other_side = math.sqrt(one_side - side_sq)

    if height_width == 0:
        width = ws * side
        height = other_side * hs
    elif height_width==1:
        width = ws * other_side
        height = hs * side
    else:
        print("something went wrong")
    T2=translate(width, height, 240,320)
    angle = angle_find_vectors(x, y, width, height)
    return angle


# height width
def angle_find(x, y, height, width):
    T = translate(x, y, 240, 320)
    angle_clock = 1.5708
    angle_counter = 1.5708
    angle_temp = [1.5708, 1.5708, 1.5708, 1.5708]

    q = which_quadrant(x, y)
    if q == 1:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, 1, 1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, -1, -1, 1)
        angle_temp[3] = angle_find2(x, y, width, 1, -1, 0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    elif q == 2:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, 1, 1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, 1, 1, 1)
        angle_temp[3] = angle_find2(x, y, width, 1, -1, 0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    elif q == 3:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, -1, -1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, -1, -1, 1)
        angle_temp[3] = angle_find2(x, y, width, -1, 1,0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    elif q == 4:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, -1, -1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, 1, 1, 1)
        angle_temp[3] = angle_find2(x, y, width, -1, 1, 0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    return angle_clock, angle_counter


def which_quadrant(x, y):
    if x <= 0:
        if y > 0:
            return 1
        else:
            return 3
    if x < 0:
        if y > 0:
            return 2
        else:
            return 4
    else:
        return 1


def max_rotation(xmin, ymin, xmax, ymax, angle, height, width):
    # make all points relative to the center of rotation
    center_x = width / 2
    center_y = height / 2

    # rightsideup
    ymin0 = height - ymin
    ymax0 = height - ymax

    xmin_r = xmin - center_x
    ymin_r = ymin0 - center_y
    xmax_r = xmax - center_x
    ymax_r = ymax0 - center_y

    # create four points
    x1 = xmin_r
    x2 = xmin_r
    x3 = xmax_r
    x4 = xmax_r

    y1 = ymin_r
    y2 = ymax_r
    y3 = ymin_r
    y4 = ymax_r

    angle_clock = 1.5708
    angle_counter = 1.5708

    temp_clock, temp_counter = angle_find(x1, y1, center_y, center_x)
    angle_clock1 = min(angle_clock, temp_clock)
    angle_counter1 = min(angle_counter, temp_counter)

    temp_clock, temp_counter = angle_find(x2, y2, center_y, center_x)
    angle_clock2 = min(angle_clock1, temp_clock)
    angle_counter2 = min(angle_counter1, temp_counter)

    temp_clock, temp_counter = angle_find(x3, y3, center_y, center_x)
    angle_clock3 = min(angle_clock2, temp_clock)
    angle_counter3 = min(angle_counter2, temp_counter)

    temp_clock, temp_counter = angle_find(x4, y4, center_y, center_x)
    angle_clock4 = min(angle_clock3, temp_clock)
    angle_counter4 = min(angle_counter3, temp_counter)

    # convert from radians to degrees
    angle_clock_con = ((180 / 3.141592) * angle_clock4)
    angle_counter_con = ((-180 / 3.141592) * angle_counter4)

    angle_clock_i = int(angle_clock_con)
    angle_counter_i = int(angle_counter_con)

    return angle_clock_i, angle_counter_i


def rotation_transform(x, y, radian):
    y_transform = y * math.cos(radian) +-x * math.sin(radian)
    x_transform = y * math.sin(radian) + x * math.cos(radian)
    val = [x_transform, y_transform]
    return x_transform, y_transform


# finds new points after rotation transformation
def post_rotation_points(xmin, ymin, xmax, ymax, angle, height, width):
    center_x = width / 2
    center_y = height / 2

    # compensate for y
    ymin0 = height - ymin
    ymax0 = height - ymax
    # make all points relative to the center of rotation
    xmin_r = xmin - center_x
    ymin_r = ymin0 - center_y
    xmax_r = xmax - center_x
    ymax_r = ymax0 - center_y

    # create four points
    x1 = xmin_r
    x2 = xmin_r
    x3 = xmax_r
    x4 = xmax_r

    y1 = ymin_r
    y2 = ymax_r
    y3 = ymin_r
    y4 = ymax_r

    # convert the angle into radians
    radian = angle * (math.pi / 180)
    x1_r, y1_r, = rotation_transform(x1, y1, radian)
    x2_r, y2_r, = rotation_transform(x2, y2, radian)
    x3_r, y3_r, = rotation_transform(x3, y3, radian)
    x4_r, y4_r, = rotation_transform(x4, y4, radian)

    # compute the new points

    # take away relative position of center
    x1_a = x1_r + center_x
    x2_a = x2_r + center_x
    x3_a = x3_r + center_x
    x4_a = x4_r + center_x

    y1_a = y1_r + center_y
    y2_a = y2_r + center_y
    y3_a = y3_r + center_y
    y4_a = y4_r + center_y


    # recompensate the ys
    y1_b = -1*(y1_a - height)
    y2_b = -1*(y2_a - height)
    y3_b = -1*(y3_a - height)
    y4_b = -1*(y4_a - height)

    p1=[x1_a,y1_b]
    p2 =[x2_a,y2_b]
    p3=[x3_a,y3_b]
    p4=[x4_a,y4_b]
    P=[p1,p2,p3,p4]

    xmin_a = int(min(x1_a, x2_a, x3_a, x4_a))
    ymin_a = int(min(y1_b, y2_b, y3_b, y4_b))
    xmax_a = int(max(x1_a, x2_a, x3_a, x4_a))
    ymax_a = int(max(y1_b, y2_b, y3_b, y4_b))
    out=[xmin_a,ymin_a,xmax_a,ymax_a]

    # return new values
    return xmin_a, ymin_a, xmax_a, ymax_a



# determines the zoom limit for a point to not be zoomed out
def zoom_point_limit(point, max_point):
    limit = .1
    distance_max = 1
    centerpoint = int(max_point / 2)
    # distance from center=0
    distance_center = abs(point - centerpoint)

    # putting in a certain tolerance point
    if limit < .2:
        limit = .2

    limit = (distance_center / centerpoint)
    return limit


# determines points after zoom
def new_zoom_points(point, zoom_level, max_point):
    distance_max = 1
    new_point = 1
    center_point = int(max_point / 2)
    # distance from center
    distance_center = abs(point - center_point)
    # if point is bellow the center
    if point < center_point:
        new_point = center_point - int((distance_center / zoom_level))
    # if the point is above center
    elif point > center_point:
        new_point = center_point + int((distance_center / zoom_level))
    else:
        new_point = center_point + int((distance_center / zoom_level))
    return new_point


def display(shift_img, xmin_shift, ymin_shift, xmax_shift, ymax_shift):
    xmin_shift=int(xmin_shift)
    ymin_shift=int(ymin_shift)
    xmax_shift=int(xmax_shift)
    ymax_shift=int(ymax_shift)
    S = np.shape(shift_img)
    height = S[0]
    width = S[1]


    U=np.zeros((height,width,3))



    U[0:height,0:width,0:3]=shift_img[0:height,0:width,0:3]


    U[ymin_shift, xmin_shift] = [255, 255, 255]
    U[ymax_shift, xmax_shift] = [255, 255, 255]

    U[ymin_shift + 1, xmin_shift] = [255, 255, 255]
    U[ymax_shift + 1, xmax_shift] = [255, 255, 255]

    U[ymin_shift, xmin_shift + 1] = [255, 255, 255]
    U[ymax_shift, xmax_shift + 1] = [255, 255, 255]

    U[ymin_shift - 1, xmin_shift] = [255, 255, 255]
    U[ymax_shift - 1, xmax_shift] = [255, 255, 255]

    U[ymin_shift, xmin_shift - 1] = [255, 255, 255]
    U[ymax_shift, xmax_shift - 1] = [255, 255, 255]
    U = np.uint8(U)
    UU = Image.fromarray(U)
    UU.show()


def cutoff(img_arr, xmin, ymin, xmax, ymax):



    S = np.shape(img_arr)
    x_diff=xmax-xmin
    y_diff=ymax-ymin

    height = S[0]
    width = S[1]

    # take away chunk from pic
    img_chunked = np.zeros((height, width, 3))
    img_chunked[0:height,0:width]=img_arr[0:height,0:width]
    blank = np.zeros((ymax - ymin, xmax - xmin, 3))
    # blank out covered area
    img_chunked[ymin:ymax, xmin: xmax] = blank

    # take a random chunk from chunked
    y_limit=int(height-(ymax))
    x_limit=int(width-(xmax))

    random_y = randint(1, y_limit)
    random_x = randint(1, x_limit)

    random_chunk = img_chunked[random_y+ymin:random_y + ymax, random_x+xmin:random_x + xmax]
    # insert chunk in missing piece
    A= img_chunked[ymin:ymax, xmin: xmax]
    B= random_chunk
    img_chunked[ymin:ymax, xmin: xmax] = random_chunk
    img_chunked = np.uint8(img_chunked)
    # return new image
    return img_chunked
#fic any pacularities in data
def problem_points(xmin, ymin, xmax, ymax,height,width):
    xm=xmin+0
    ym=ymin+0
    xmx=xmax+0
    ymx=ymax+0

    problem=False
    #check if negative
    if xmin<0:
        xmin=0
        problem=True
    if ymin < 0:
        ymin = 0
        problem = True
    if xmax<0:
        xmax=0
        problem = True
    if ymax < 0:
        ymax = 0
        problem = True

    #check if over
    if xmin>=width:
        xmin=width-1
        problem = True
    if ymin >= height:
        ymin=height-1
        problem = True
    if xmax>=width:
        xmax=width-1
        problem = True
    if ymax >= height:
        ymax=height-1
        problem = True


    #check if any are equal
    #convert to integers
    xmin_int=int(xmin)
    ymin_int=int(ymin)
    xmax_int=int(xmax)
    ymax_int=int(ymax)

    if xmin_int==xmax_int:
        xmin=0
        xmax=width-1
        problem = True

    if ymin_int==ymax_int:
        ymin=0
        ymax=height-1
        problem = True

    if problem:
        print("issue with points", str(xm), str(ym), str(xmx), str(ymx))
        print("fixed to", str(xmin), str(ymin), str(xmax), str(ymax))

    return xmin, ymin, xmax, ymax


def resize_and_points(img_arr, xmin, ymin, xmax, ymax, maxsize):
    max_height=maxsize[0]
    max_width=maxsize[1]

    original_shape = np.shape(img_arr)

    original_height = original_shape[0]
    original_width = original_shape[1]

    S0 = np.zeros((original_height, original_width, 3))
    S0[0:original_height, 0:original_width, 0:3] = img_arr[0:original_height, 0:original_width, 0:3]
    S0=np.uint8(S0)
    SA = PIL.Image.fromarray(S0)
    SA.thumbnail((max_height, max_width))
    SC = np.array(SA)

    new_shape = np.shape(SC)
    new_height = new_shape[0]
    new_width = new_shape[1]

    x_scale_factor = new_width / original_width
    y_scale_factor = new_height / original_height

    # compensate the zoom points by the scale factors
    xmin_scaled = xmin * x_scale_factor
    xmax_scaled = xmax * x_scale_factor
    ymin_scaled = ymin * y_scale_factor
    ymax_scaled = ymax * y_scale_factor
    return SC, xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled

#target size has to be list
def aug(img_arr_pre, xmin, ymin, xmax, ymax,target_size):
    global datagen
    img_shape = np.shape(img_arr_pre)
    height = img_shape[0]
    width = img_shape[1]
    #process the image array
    #remove extra dimension if png file
    img_arr=img_arr_pre[0:height,0:width,0:3]


    # determine macimum zoom range
    xmin_limit = zoom_point_limit(xmin, width)
    xmax_limit = zoom_point_limit(xmax, width)
    ymin_limit = zoom_point_limit(ymin, height)
    ymax_limit = zoom_point_limit(ymax, height)

    zoom_x_limit = max(xmin_limit, xmax_limit)
    zoom_y_limit = max(ymin_limit, ymax_limit)
    # from this range choose random zoom level
    zoom_x = random.uniform(zoom_x_limit, 1.5)
    zoom_y = random.uniform(zoom_y_limit, 1.5)

    # determine new points from this zoom
    xmin_zoom = new_zoom_points(xmin, zoom_x, width)
    xmax_zoom = new_zoom_points(xmax, zoom_x, width)
    ymin_zoom = new_zoom_points(ymin, zoom_y, height)
    ymax_zoom = new_zoom_points(ymax, zoom_y, height)
    zoompoints = [xmin_zoom, ymin_zoom, xmax_zoom, ymax_zoom]
    #zoom in or out of image
    zoom_img = datagen.apply_transform(x=img_arr, transform_parameters={'zx': zoom_y, 'zy': zoom_x})
    #scale the image down to target size
    #determine the scale factors
    x_scale_factor =target_size[1]/width
    y_scale_factor =target_size[0]/height

    #compensate the zoom points by the scale factors
    xmin_scaled=xmin_zoom*x_scale_factor
    xmax_scaled=xmax_zoom*x_scale_factor
    ymin_scaled=ymin_zoom *y_scale_factor
    ymax_scaled=ymax_zoom *y_scale_factor

    #determine new height and width
    height_scaled=target_size[0]
    width_scaled=target_size[1]
    #resize image to target size
    zoom_img2 =PIL.Image.fromarray(zoom_img)
    scaled_img_0 =zoom_img2.resize((target_size[0],target_size[1]))
    scaled_img =np.array(scaled_img_0)

    #display(scaled_img,xmin_scaled,ymin_scaled,xmax_scaled,ymax_scaled)

    # rotate image
    #determine maximum range image can be rotated
    angle_clock,angle_clock_counter = max_rotation(xmin, ymin, xmax, ymax, 90, height_scaled, width_scaled)
    #draw out random number from that range
    if angle_clock-angle_clock_counter<=0:
        angle=0
    else:
        angle = randint(angle_clock_counter, angle_clock)
    #rotate image
    #scaled_seperated=np.zeros((height_scaled,width_scaled,3))
    #scaled_seperated[0:height_scaled,0:width_scaled]=scaled_img[0:height_scaled,0:width_scaled]
    rotated_img = datagen.apply_transform(x=scaled_img, transform_parameters={'theta': angle})
    #determine new points afer rotation
    xmin_rotated, ymin_rotated, xmax_rotated, ymax_rotated = post_rotation_points(xmin_scaled, ymin_scaled, xmax_scaled,
                                                          ymax_scaled, angle, height_scaled, width_scaled)
    #display(rotated_img,  xmin_rotated, ymin_rotated, xmax_rotated, ymax_rotated )

    #shift image
    #determine maximum range of shift
    right_shift_limit = width_scaled - xmax_rotated - 1
    left_shift_limit = -1 * xmin_rotated
    up_shift_limit = -1 * ymin_rotated  # height - new_ymax
    down_shift_limit = height_scaled - ymax_rotated - 1  # -1 * new_ymin

    # chose random integer from this range
    left=int(min(left_shift_limit, right_shift_limit))
    right=int(max(left_shift_limit, right_shift_limit))
    down=int(min(up_shift_limit, down_shift_limit))
    up=int(max(up_shift_limit, down_shift_limit))
    if right-left<=0:
        horizontal_shift=0
    else:
        horizontal_shift = (randint(left, right))

    if up-down <= 0:
        vertical_shift=0
    else:
        vertical_shift = -(randint(down, up))
    # = (randint(left, right))
    #vertical_shift = -(randint(down, up))

    # determine new points from these shifts
    shift_img = np.zeros((height_scaled, width_scaled, 3))
    shift_img[0:height_scaled,0:width_scaled]=rotated_img[0:height_scaled,0:width_scaled]
    #shift image
    shift_img = np.roll(shift_img, horizontal_shift, axis=1)
    shift_img = np.roll(shift_img, -vertical_shift, axis=0)
    #determine new points and truncate into integers
    xmin_shift = (xmin_rotated + horizontal_shift * 1)
    xmax_shift = (xmax_rotated + horizontal_shift * 1)
    ymin_shift = (ymin_rotated - vertical_shift * 1)
    ymax_shift = (ymax_rotated - vertical_shift * 1)

    #fix any irregularities
    xmin_shift, ymin_shift, xmax_shift, ymax_shift=problem_points(xmin_shift, ymin_shift, xmax_shift, ymax_shift,height_scaled,width_scaled)
    test_points = [xmin_shift, ymin_shift,xmax_shift, ymax_shift]
    #display(shift_img,xmin_shift, ymin_shift,xmax_shift, ymax_shift)


    #turn values in ratio of height and width
    xmin_ratio = xmin_shift/width_scaled
    xmax_ratio = xmax_shift/width_scaled
    ymin_ratio = ymin_shift/height_scaled
    ymax_ratio = ymax_shift/height_scaled
    # construct return array
    new_points=[xmin_ratio, ymin_ratio,xmax_ratio, ymax_ratio]
    #take chunk out
    img_chunked=cutoff(shift_img,xmin_shift, ymin_shift,xmax_shift, ymax_shift)

    #unsigned integers
    img_chunked=np.uint8(img_chunked)
    shift_img=np.uint8(shift_img)

    return shift_img, new_points, img_chunked

def start():
    global source
    global destination,name,augmentsnum
    #get information from text entries
    name=project_name_txt.get()
    augmentsnum=int(augnum_txt.get())
    wb = xlwt.Workbook()
    ws = wb.add_sheet('data')
    destination = destination + name + "_augmented_images/"
    #ensure the destination is there
    if not os.path.isdir(destination):
        os.mkdir(destination)


    target_width = 224
    target_height = 224

    #start rowcout at one after the excel labels at the top
    rowcount=1

    ws.write(0, 0, "ID")
    ws.write(0, 1, "xmin")
    ws.write(0, 2, "ymin")
    ws.write(0, 3, "xmax")
    ws.write(0, 4, "ymax")
    ws.write(0, 5, "P")

    ws.write(0, 6, "xmin scaled back")
    ws.write(0, 7, "ymin scaled back")
    ws.write(0, 8, "xmax scaled back")
    ws.write(0, 9, "ymax scaled back")

    ws.write(0, 11, "target height" + str(target_height))
    ws.write(0, 12, "traget width" + str(target_width))

    df = pd.read_excel(source)
    df_list = df.to_numpy()
    file_list = np.delete(df_list, 0, 0)

    shape = np.shape(file_list)
    rows = shape[0]


    for i in range(rows):

        file_path=file_list[i, 0]

        xmin = int(file_list[i, 1])
        ymin = int(file_list[i, 2])
        xmax = int(file_list[i, 3])
        ymax = int(file_list[i, 4])
        im = PIL.Image.open(file_path)
        img_arr = np.array(im)
        # truncate the 4th dimension if png
        image_shape = np.shape(img_arr)
        img_height = image_shape[0]
        img_width = image_shape[1]
        img_arr2 = img_arr[0:img_height, 0:img_width, 0:3]

        SC, xmin_s, ymin_s, xmax_s, ymax_s = resize_and_points(img_arr2, xmin, ymin, xmax, ymax, [600, 600])



        for j in range(augmentsnum):
            new = aug(SC, xmin_s, ymin_s, xmax_s, ymax_s,[224,224])
            new_img = new[0]
            new_points = new[1]
            chunked_img=new[2]

            xmin2 = new_points[0]
            ymin2 = new_points[1]
            xmax2 = new_points[2]
            ymax2 = new_points[3]

            xmin_scaled = xmin2
            ymin_scaled = ymin2
            xmax_scaled = xmax2
            ymax_scaled = ymax2

            if xmin_scaled <= 0:
                xmin_scaled = 0
            if ymin_scaled <= 0:
                ymin_scaled = 0
            if xmax_scaled > 1:
                xmax_scaled = .99
            if ymax_scaled > 1:
                ymax_scaled = .99

            ws.write(rowcount, 0, str(destination + "1/"+str(rowcount) + ".jpeg"))
            ws.write(rowcount, 1, xmin_scaled)
            ws.write(rowcount, 2, ymin_scaled)
            ws.write(rowcount, 3, xmax_scaled)
            ws.write(rowcount, 4, ymax_scaled)
            ws.write(rowcount, 5, 0)

            xmin_scaled_back = int(xmin_scaled * 224) -1
            ymin_scaled_back = int(ymin_scaled * 224)-1
            xmax_scaled_back = int(xmax_scaled * 224)-1
            ymax_scaled_back = int(ymax_scaled * 224)-1

            ws.write(rowcount, 6, xmin_scaled_back)
            ws.write(rowcount, 7, ymin_scaled_back)
            ws.write(rowcount, 8, xmax_scaled_back)
            ws.write(rowcount, 9, ymax_scaled_back)
            new_i = PIL.Image.fromarray(new_img)
            new_c=PIL.Image.fromarray(chunked_img)
            # new_i.show()
            new_im = new_i.resize((224, 224))
            #new_cim= new_c.resize((224, 224))
            U = np.array(new_im)
            #C = np.array(new_cim)


            UU = PIL.Image.fromarray(U)
            #CC=  Image.fromarray(C)
            #UU.save(destination + "1/"+str(rowcount) + ".jpeg")
            #CC.save(destination + "0/"+str(rowcount) + ".jpeg")
            UU.save(destination  + str(rowcount) + ".jpeg")
            rowcount = rowcount + 1



        print(str(i))

    wb.save(destination+name+".xls")


def assign_source():
    global source,source_set,destination_set
    source=filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("xls","*.xls"),("all files","*.*")))
    source_set=True
    if destination_set and source_set:
        start_btn.configure(state='normal')
    source_lbl.configure(text=source)


def assign_destination():
    global destination,source_set,destination_set
    dest_String = filedialog.askdirectory()
    destination = dest_String + "/"
    destination_set=True
    if destination_set and source_set:
        start_btn.configure(state='normal')
    destination_lbl.configure(text=destination)

project_name_lbl=Label(master=file_frame,text="Project name")
project_name_lbl.grid(column=0,row=0,padx=5,pady=5)
project_name_txt=Entry(master=file_frame)
project_name_txt.grid(column=1,row=0,padx=5,pady=5)



source_btn=Button(master=file_frame,command=assign_source,text="source xls file")
source_btn.grid(column=0,row=1,padx=5,pady=5)

destination_btn=Button(master=file_frame,command=assign_destination,text="save directory")
destination_btn.grid(column=0,row=2,padx=5,pady=5)

source_lbl=Label(master=file_frame)
source_lbl.grid(column=1,row=1,padx=5,pady=5)

destination_lbl=Label(master=file_frame)
destination_lbl.grid(column=1,row=2,padx=5,pady=5)


start_btn=Button(master=file_frame,command=start,text="start")
start_btn.grid(column=0,row=4,padx=5,pady=5)
start_btn.configure(state='disabled')

augnum_lbl=Label(master=file_frame,text="number of augmentations per picture")
augnum_lbl.grid(column=0,row=3,padx=5,pady=5)

augnum_txt=Entry(master=file_frame)
augnum_txt.grid(column=1,row=3,padx=5,pady=5)
augnum_txt.insert(END,"5")

window.mainloop()
