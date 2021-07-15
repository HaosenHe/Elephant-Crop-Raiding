'''
Haosen He 2020
Wake Forest University
This is the CRM package for map/walk visualization and block classification.
'''
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from matplotlib import colors
import pandas as pd
from collections import defaultdict
import random


def RGB2HEX(color):
    '''
    Returns the HEX code given the RGB code.

    color -- color code in RGB
    '''
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    '''
    Read image and convert to RGB color

    image_path -- path of the image
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_colors(img):
    '''
    Displays the colors in the image (img) in a pie chart.

    img -- image input
    '''
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.show()


def slicer(img, row, col):
    '''
    Slice an image of size x*y into row*col number of subimages of size (x//col)*(y//row)
    and return the subimages in a row*col sized 2D array.

    img -- the image to slice
    row -- number of rows in the output matrix
    col -- number of columns in the output matrix

    Illustration:
    Splitting an image into 2*2=4 subimages:      __              __      __               __
   __                           __                |                |      |                 |
   |／￣乀乀＼　                  |                    ／￣乀乀＼
    |　ー　ー \　 ／￣￣￣￣￣￣＼                      |　ー　ー \　           ／￣￣￣￣￣￣＼
    |　 ◉　◉ |  | Jiaran Diana\                  |_  |　 ◉　◉ |  _|      |_|Jiaran Diana \_|
    \　　 ▱　/ /_ is the best!/   ==========>     __              __      __               __
     ＼　　 イ　   \          /                    |   \　　 ▱　/ /_|      |  is the best!/  |
     ／　　　\      ￣￣￣￣￣                           ＼　　 イ　            \          /
   |/　|　　  \                   |                     ／　　　\                ￣￣￣￣￣
   ￣                           ￣                |_   /　|　　   \  _|    |_               _|
    Return the subimages in a 2D array.
    '''
    dx = img.shape[1]//col
    dy = img.shape[0]//row
    imgls = []
    for i in range(0, row):
        for j in range(0, col):
            imgls.append(img[i*dy:(i+1)*dy, j*dx:(j+1)*dx])
    output = []
    for i in range(0, row):
        output.append([0]*col)
    for i in range(0, row):
        for j in range(0, col):
            output[i][j] = imgls[i*col+j]
    return output


def identify_land(image):
    '''
    Identify the block type of a land in the subimage

    image -- subimage to identify

    Return 0 if the land is a water block
           1 if the land is a tree block
           2 if the land is a crop block
    '''
    height = len(image)
    width = len(image[0])
    water1 = '#0a47c9'
    tree1 = '#58be4b'
    tree2 = '#00a002'
    crop = '#f9fa64'
    colordict = defaultdict(int)
    for i in image:
        for j in i:
            colordict[RGB2HEX(j)] += 1
    wratio = (colordict[water1])/(height*width)
    tratio = (colordict[tree1]+colordict[tree2])/(height*width)
    cratio = colordict[crop]/(height*width)
    if wratio > 0.01:
        return 0
    elif cratio > 0.2:
        return 2
    elif tratio > 0.5:
        return 1


def Process(image_path, row, col):
    '''
    Convert the image into a coarser-resolution map(lattice) with resolution row*col

    image_path -- path of the image file
    row -- number of rows
    col -- number of columns

    Return the set of crop blocks, the set of water blocks, and the set of tree blocks.
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    subimages = slicer(image, row, col)
    waterset = set()
    cropset = set()
    treeset = set()
    for i in range(0, len(subimages)):
        for j in range(0, len(subimages[i])):
            if identify_land(subimages[i][j]) == 2:
                cropset.add((i, j))
            elif identify_land(subimages[i][j]) == 0:
                waterset.add((i, j))
            elif identify_land(subimages[i][j]) == 1:
                treeset.add((i, j))
    return cropset, waterset, treeset


def generate_map(rows, cols, cropset=None, waterset=None, treeset=None, protectset=None):
    '''
    Generate a map given the size and resources information

    rows, cols -- size information (size = rows*cols)
    cropset -- the set of crop blocks in the map
    waterset -- the set of water blocks in the map
    treeset -- the set of trees in the map
    protectset -- the set of protected blocks in the map
    '''
    arr = []
    for i in range(cols):
        col = []
        for j in range(rows):
            col.append(0)
        arr.append(col)
    if cropset != None:
        for i in cropset:
            arr[i[0]][i[1]] = 3
    if waterset != None:
        for i in waterset:
            arr[i[0]][i[1]] = 2
    if treeset != None:
        for i in treeset:
            arr[i[0]][i[1]] = 1
    return np.array(arr)


def plot_map(row, col, cropset, waterset, treeset):
    '''
    Plot the lattice map given size and resource information

    rows, cols -- size information (size = rows*cols)
    cropset -- the set of crop blocks in the map
    waterset -- the set of water blocks in the map
    treeset -- the set of trees in the map

    Return a lattice map with blue indicating water, orange
    indicating crop, and green indicating tree. White regions
    are thus savanna.
    '''
    data = generate_map(row, col, cropset, waterset, treeset)
    # create a discrete colormap
    cmap = colors.ListedColormap(['white', 'green', 'blue', 'orange'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(data, cmap=cmap, norm=norm)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(0, col, 1))
    ax.set_yticks(np.arange(0, row, 1))
    plt.show()


def binary_search(arr, x):
    '''
    A standard binary search algorithm used to detect potential errors.

    arr -- array of small land blocks
    x -- size in one of the dimension
    '''
    l = 0
    r = len(arr)-1
    while l <= r:
        mid = l+(r-l)//2
        if arr[mid] <= x and arr[mid+1] >= x:
            return mid
        elif arr[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    print('ERROR, Cannot discretize location {}!YGIUY'.format(x))


def discretize_walk(X, Y, x, y):
    '''
    Discretize a walk recorded by GPS collar data and fit it into the lattice.
    This function need to be generalized when it is applied to other projects.

    X -- a list/array/series of longitudinal data indicating animal location on the Earth
    Y -- a list/array/series of latitudinal data
    x -- the number of rows in the lattice
    y -- the number of columns in the lattice

    Return two np.arrays of discretized longitude (column number) and latitude
    (row number) on the lattice.
    '''
    dx = (34.7-33.96)/x
    dy = (19.3-18.6)/y
    xtick = np.arange(33.96, 34.7, dx)
    ytick = np.arange(-19.3, -18.6, dy)
    dscrt_X = np.array([])
    dscrt_Y = np.array([])
    for i in X:
        dscrt_X = np.append(dscrt_X, binary_search(xtick, i))
    for j in Y:
        dscrt_Y = np.append(dscrt_Y, binary_search(ytick, j))
    return dscrt_X, dscrt_Y


def plot_walk(row, col, cropset, waterset, treeset, walk):
    '''
    Plot a discrete walk.

    rows, cols -- size information (size = rows*cols)
    cropset -- the set of crop blocks in the map
    waterset -- the set of water blocks in the map
    treeset -- the set of trees in the map
    walk -- a list of 2-tuples indicating discrete location

    Return a lattice map with blue indicating water, orange
    indicating crop, and green indicating tree. White regions
    are thus savanna.
    '''
    # plot one walk in a discretized map
    data = generate_map(row, col, cropset, waterset, treeset)
    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'green', 'blue', 'orange'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(data, cmap=cmap, norm=norm)
    xs = []
    ys = []
    for i in walk:
        xs.append(i[0])
        ys.append(i[1])
    ax.plot(ys, xs, linewidth=5, color='r')
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(0, col, 1))
    ax.set_yticks(np.arange(0, row, 1))
    plt.show()


def plot_walks(row, col, cropset, waterset, treeset, walkset):
    '''
    Plot a set of discrete walks.

    rows, cols -- size information (size = rows*cols)
    cropset -- the set of crop blocks in the map
    waterset -- the set of water blocks in the map
    treeset -- the set of trees in the map
    walkset -- a set containing lists of 2-tuples indicating discrete location

    Return a lattice map with blue indicating water, orange
    indicating crop, and green indicating tree. White regions
    are thus savanna.
    '''
    data = generate_map(row, col, cropset, waterset, treeset)
    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'green', 'blue', 'orange'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(data, cmap=cmap, norm=norm)
    for walk in walkset:
        xs = []
        ys = []
        for i in walk:
            xs.append(i[0])
            ys.append(i[1])
        ax.plot(ys, xs, linewidth=5, color='r')
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(0, col, 1))
    ax.set_yticks(np.arange(0, row, 1))
    plt.show()


'''
def compare_walks(row, col, cropset, waterset, treeset, walkset1, walk2, walk3, walkls4):
    data = generate_map(row, col, cropset, waterset, treeset)
    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'green', 'blue', 'orange'])
    bounds = [0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(data, cmap=cmap, norm=norm)
    for walk in walkset1:
        xs=[]
        ys=[]
        for i in walk:
            xs.append(i[0])
            ys.append(i[1])
        ax.plot(ys,xs,linewidth=5,color='r', alpha=0.8)
    xs2=[]
    ys2=[]
    for i in walk2:
        xs2.append(i[0])
        ys2.append(i[1])
    ax.plot(ys2,xs2,linewidth=5,color='purple', alpha=0.8)
    xs3=[]
    ys3=[]
    for i in walk3:
        xs3.append(i[0])
        ys3.append(i[1])
    ax.plot(ys3,xs3,linewidth=5,color='k', alpha=0.8)
    for walk in walkls4:
        xs=[]
        ys=[]
        for i in walk:
            xs.append(i[0])
            ys.append(i[1])
        ax.plot(ys,xs,linewidth=5,color='c', alpha=0.5)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(0, col, 1));
    ax.set_yticks(np.arange(0, row, 1));
    plt.show()
'''
