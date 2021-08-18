# Creating-art-using-Genetic-algorithms

## Kamil Sabbagh

##### Discrebtion:
In this task we are creating original art using AI and specifically genetic algorithms. A
512*512 pixels input image will be given and the aim is to create a program that will
create an art using the given picture, various ways are applicable such as recreating
the image using geometric shapes like triangle, circles, etc .. or using other images
as a basis of the new image.
My approach was creating a mosaic from the given input and adding digital noise to
the image, giving it a digital touch. Each piece in the mosaic shares the same (RGB)
values.

##### Implementation of the algorithm:
1- Creating an initial population (Gen #0) with a 100 image. Each image consists of
random blocks of pixels sharing the same random color.
2- Evaluating all the individuals by giving them the error score using the fitness
function mentioned before.
3- Sorting the current population based on the error scores
4- the top ten images (10% of the total population) will be selected for breading
5- two random images ( from the top 10% selected individuals ) will be chosen for
breading using the crossover function and new offspring will be adding the new
population

#### Illustration (Examples):
[AI genetic algorithms report (Original).pdf](https://github.com/Kamil-Sabbagh/Creating-art-using-Genetic-algorithms/files/7005992/AI.genetic.algorithms.report.Original.pdf)
[AI genetic algorithms report (AI generated).pdf](https://github.com/Kamil-Sabbagh/Creating-art-using-Genetic-algorithms/files/7005993/AI.genetic.algorithms.report.AI.generated.pdf)


##### How to use:
To use this program all you need is a working python IDE, then:
1) open the folder, and add the the image you want to edit to the folder
2) open main.py using your prefered python IDE. 
3) change the imgage name in the code in line #16 to the image name you have add
4) run the code and wait for your image to be genereated
