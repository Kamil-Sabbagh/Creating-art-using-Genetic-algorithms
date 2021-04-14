import cv2 as cv
import numpy
import time
import numpy as np
from random import choices
from random import randint, random
from numba import njit, int64, float64
import random
from PIL import Image

# original image:
from numba import types
from numba.experimental import jitclass

#first we open the image and resize it if needed
original = Image.open("head.jpg")
original = original.resize((512, 512))
original.thumbnail((512, 512))
original = np.array(original)

#the zein factor the area of the block this can be change to get some different block size
zein_factor = 2

#all the jitclass are from the numba library they are being used for optimizing the code
#they can be ignored while reading as they are just for optimizing
@jitclass([('rgb', types.List(float64))])
class Count:
    count: int
    rgb: list

    def __init__(self, count, rgb):
        self.count = count
        self.rgb = rgb


@jitclass([('rgb', types.List(float64))])
class RGB:
    rgb: list

    def __init__(self, rgb):
        self.rgb = rgb


# I choose to manipulate photos as numpay arrays with a representative pixel
# follows as the implementation of that for the original picture
@njit
def original_to_array(original):
    original_mosaic_colors = {}
    original_mosaic_counts = {}
    print(zein_factor)
    for r in range(0, 512, zein_factor):
        for c in range(0, 512, zein_factor):
            i = r // zein_factor
            j = c // zein_factor
            if i in original_mosaic_colors and j in original_mosaic_colors[i]:
                original_mosaic_counts[i][j].count += 1
                original_mosaic_counts[i][j].rgb[0] += float(original[r][c][0])
                original_mosaic_counts[i][j].rgb[1] += float(original[r][c][1])
                original_mosaic_counts[i][j].rgb[2] += float(original[r][c][2])
                original_mosaic_colors[i][j] = RGB([
                    original_mosaic_counts[i][j].rgb[0] / original_mosaic_counts[i][j].count,
                    original_mosaic_counts[i][j].rgb[1] / original_mosaic_counts[i][j].count,
                    original_mosaic_counts[i][j].rgb[2] / original_mosaic_counts[i][j].count])
            else:
                if i not in original_mosaic_colors:
                    original_mosaic_counts[i] = {j: Count(1, [float(original[r][c][0]), float(original[r][c][1]), float(original[r][c][2])])}
                    original_mosaic_colors[i] = {j: RGB([
                        original_mosaic_counts[i][j].rgb[0] / original_mosaic_counts[i][j].count,
                        original_mosaic_counts[i][j].rgb[1] / original_mosaic_counts[i][j].count,
                        original_mosaic_counts[i][j].rgb[2] / original_mosaic_counts[i][j].count])}
                else:
                    original_mosaic_counts[i][j] = Count(1, [float(original[r][c][0]), float(original[r][c][1]), float(original[r][c][2])])
                    original_mosaic_colors[i][j] = RGB([
                        original_mosaic_counts[i][j].rgb[0] / original_mosaic_counts[i][j].count,
                        original_mosaic_counts[i][j].rgb[1] / original_mosaic_counts[i][j].count,
                        original_mosaic_counts[i][j].rgb[2] / original_mosaic_counts[i][j].count])
    return original_mosaic_colors


original_mosaic_colors = original_to_array(np.array(original))

#follows the implementaion for the fitness function
@njit
def fitness(original_mosaic_colors, temp):
    score = 0
    for i in range(512 // zein_factor):
        for j in range(512 // zein_factor):
            if i in original_mosaic_colors and j in original_mosaic_colors[i]:
                for cc in range(3):
                    score += abs(original_mosaic_colors[i][j].rgb[cc] - temp[i][j][cc])
    return score



# this function is used to color each block with the representative colors and return an image
def present(x):
    img1 = np.array(x)
    img1_mosaic_counts = compute_mosaic_counts(img1)
    for r in range(0, 512):
        for c in range(0, 512):
            i = r // zein_factor
            j = c // zein_factor
            if i in img1_mosaic_counts and j in img1_mosaic_counts[i]:
                for color in range(3):
                    img1[r][c][color] = img1_mosaic_counts[i][j].rgb[color]
    temp = Image.fromarray(img1, 'RGB')
    temp.show()
    return


#the following function is used to create the representative numpay array representations for images from the population
@njit
def compute_mosaic_counts(x):
    img1 = x
    img1_mosaic_colors = {}
    img1_mosaic_counts = {}
    for r in range(0, 512, zein_factor):
        for c in range(0, 512, zein_factor):
            xx = r // zein_factor
            yy = c // zein_factor
            if xx in img1_mosaic_colors and yy in img1_mosaic_colors[xx]:
                img1_mosaic_counts[xx][yy].count += 1
                img1_mosaic_counts[xx][yy].rgb[0] += float(img1[r][c][0])
                img1_mosaic_counts[xx][yy].rgb[1] += float(img1[r][c][1])
                img1_mosaic_counts[xx][yy].rgb[2] += float(img1[r][c][2])
                img1_mosaic_colors[xx][yy] = RGB([img1_mosaic_counts[xx][yy].rgb[0] / img1_mosaic_counts[xx][yy].count,
                                                  img1_mosaic_counts[xx][yy].rgb[1] / img1_mosaic_counts[xx][yy].count,
                                                  img1_mosaic_counts[xx][yy].rgb[2] / img1_mosaic_counts[xx][yy].count])

            else:
                if xx not in img1_mosaic_counts:
                    img1_mosaic_counts[xx] = {yy: Count(1, [float(img1[r][c][0]), float(img1[r][c][1]), float(img1[r][c][2])])}
                    img1_mosaic_colors[xx] = {yy: RGB([img1_mosaic_counts[xx][yy].rgb[0] / img1_mosaic_counts[xx][yy].count,
                                                      img1_mosaic_counts[xx][yy].rgb[1] / img1_mosaic_counts[xx][yy].count,
                                                      img1_mosaic_counts[xx][yy].rgb[2] / img1_mosaic_counts[xx][yy].count])}
                else:
                    img1_mosaic_counts[xx][yy] = Count(1, [float(img1[r][c][0]), float(img1[r][c][1]), float(img1[r][c][2])])
                    img1_mosaic_colors[xx][yy] = RGB([img1_mosaic_counts[xx][yy].rgb[0] / img1_mosaic_counts[xx][yy].count,
                                                      img1_mosaic_counts[xx][yy].rgb[1] / img1_mosaic_counts[xx][yy].count,
                                                      img1_mosaic_counts[xx][yy].rgb[2] / img1_mosaic_counts[xx][yy].count])
    return img1_mosaic_counts


# here is the implementation of the crossover function
@njit
def crossover(original_mosaic_colors, a, b):
    temp = np.zeros((512, 512, 3), dtype=numpy.float64)
    for r in range(0, 512, zein_factor):
        for c in range(0, 512, zein_factor):
            i = r // zein_factor
            j = c // zein_factor
            color = randint(0, 2)
            # original_mosaic_colors[(i,j)][cc] -  temp_mosaic_colors[(i,j)][cc]
            if randint(0, 100) < 3 :
                something_random = randint(0, 5)
                temp[r][c][color] = float(original_mosaic_colors[i][j].rgb[color]) + randint((-1) * something_random,
                                                                                      something_random)
            elif abs(a[r][c][color] - original_mosaic_colors[i][j].rgb[color]) < abs(
                    b[r][c][color] - original_mosaic_colors[i][j].rgb[color]):
                temp[r][c] = a[r][c]
            else:
                temp[r][c] = b[r][c]


    return temp


# first Generation :
population = []

#creating the first generation

@njit
def gen_first(im):
    blocks = {}
    for r in range(0, 512, zein_factor):
        for c in range(0, 512, zein_factor):
            i = r // zein_factor
            j = c // zein_factor
            if i in blocks and j in blocks[i]:
                im[r][c] = blocks[i][j]
            else:
                re = random.randint(0, 255)
                gr = random.randint(0, 255)
                bl = random.randint(0, 255)
                if i not in blocks:
                    blocks[i] = {j: (re, gr, bl)}
                else:
                    blocks[i][j] = (re, gr, bl)
                im[r][c] = (re, gr, bl)

prev_ans = 99999999999999

#creatingt he first generation
for _ in range(100):
    im = Image.new("RGB", (512, 512))
    im = np.array(im)

    gen_first(im)

    img = Image.fromarray(im, 'RGB')
    population.append(img)
start = time.time()


# the algorithm will run for a 100 generation or until score is changing insignificantly

for gen in range(100):
    #we need the ranked_solutions to keep track of the best Chromosomes in the generation
    ranked_solutions = []

    #we calculate the fitness for every chromosome
    for x in population:
        ranked_solutions.append((fitness(original_mosaic_colors, np.array(x)), x))
    #then we sort them based on the fitness
    ranked_solutions.sort(key=lambda tup: tup[0])

    print(f"Lowest error score in gen {gen} is { ranked_solutions[0][0] } ")

    #this is the braking clause when the change is insignificant
    if abs(ranked_solutions[0][0] - prev_ans) < 100 :
         break

    prev_ans = ranked_solutions[0][0]
    best_solutions = ranked_solutions[:10]

    New_gen = []
    # here we populate the new generations using the crossover function
    for i in range(100):
        img1 = np.array(choices(best_solutions)[0][1], dtype=numpy.float64)
        img2 = np.array(choices(best_solutions)[0][1], dtype=numpy.float64)

        something = Image.fromarray(np.array(crossover(original_mosaic_colors, img1, img2), dtype=numpy.uint8))
        New_gen.append(something)

    population = New_gen

present(ranked_solutions[0][1])
print("searching is over!")
print("Done!")
print(time.time() - start)



