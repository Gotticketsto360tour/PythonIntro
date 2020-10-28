'''
TASKS: ALL
___________
Description:
Document for testing all functions from the module.
'''

import math_helper as M
import network_helper as NH
import read_write_helper as RW
import plots_helper as P
### FIRST GROUP:
#__________________

## TASK B)

filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}

labels = RW.read_labels(filename['images'])
M.dim(labels)

## TASK C)

filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}

images = RW.read_image(filename['images'])
M.dim(images)

## TASK D)

index_list = [5, 6, 12, 16, 20, 100, 200, 1000, 1500, 9999]
filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}
images = RW.read_image(filename['images'])
filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = RW.read_labels(filename['images'])

P.plot_images(images, labels, index_list, columns = 5) #optional argument
P.plot_images(images, labels) #without the optionals.

#Check flexibility

index_list = 15
P.plot_images(images, labels, index_list, columns = 6) #optional argument


### SECOND GROUP:
#__________________

## TASK F)

filename = "mnist_linear.weights"

json_string = RW.linear_load(filename)

RW.linear_save('json_test', json_string)

## TASK G)

M.dim(RW.image_to_vector(images[2]))

## TASK H

list_matrix = [[1,2,3], [4,5,6]]
list_matrix2 = [[2,3],[4,5], [6,7]]
list_matrix3 = [[2,3,4], [5, 6,7]]

# Dim:

M.dim(list_matrix)
M.dim(list_matrix2)
M.dim([1,2,3]) #special case for a list of only one dimension

# Add and sub:

#CHECK EXAMPLES
M.add(list_matrix, list_matrix3)
M.sub(list_matrix3, list_matrix)

#CHECK ASSUMPTIONS
M.add(list_matrix, list_matrix2)
M.sub(list_matrix, list_matrix2)

# Scalar Multiplication

#CHECK EXAMPLES
M.scalar_multiplication(list_matrix, 5)

#CHECK ASSUMPTIONS
M.scalar_multiplication(list_matrix, list_matrix2)

# Matrix multiplication:

#CHECK EXAMPLES

M.multiply(list_matrix, list_matrix2)

M.multiply(list_matrix2, list_matrix)

#CHECK ASSUMPTIONS

M.multiply(list_matrix3, list_matrix)

# Transpose:

#CHECK EXAMPLES

M.transpose(list_matrix)
M.transpose(list_matrix2)

## TASK I)

# CHECK EXAMPLE:
M.mean_square_error([1,2,3,4], [3,1,3,2]) #checks out

# CHECK ASSUMPTIONS
M.mean_square_error([1,2,3,4], 5) #checks out

## TASK J)

# CHECK EXAMPLE:
M.argmax([6, 2, 7, 10, 5]) #checks out

# CHECK ASSUMPTIONSS
M.argmax(3) #checks out

## TASK K)

# CHECK EXAMPLE:
M.categorical(3) #checks out

## TASK L)

network = RW.linear_load('mnist_linear.weights')
images = RW.read_image('train-images.idx3-ubyte')
labels = RW.read_labels('train-labels.idx1-ubyte')
image_vector = RW.image_to_vector(images[0])

M.predict(network, image_vector)

## TASK M)

#M.evaluate(network, images, labels) - how do you test this?

## TASK N)

index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
predictions = [7, 2, 1, 3, 4, 5, 6, 6, 8, 9, 10, 11]
filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 'train-images.idx3-ubyte'}
images = RW.read_image(filename['images'])
filename = {'images' : 't10k-labels.idx1-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = RW.read_labels(filename['images'])
predictions = [7, 2, 1, 3, 4, 5, 6, 6, 8, 9, 10, 11]

P.plot_images_new(images, labels, index_list, 5, predictions)
P.plot_images_new(images, labels, 20, 5)
P.plot_images_new(images, labels)

## TASK O)

network = RW.linear_load('mnist_linear.weights')
A, b = network

P.weights_plot(A, plt_col = 5)

### THIRD GROUP:
#__________________

## TASK P)

l = NH.create_batches(list(range(8)), 3)
l

## TASK Q)
filename_test = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 't10k-labels.idx1-ubyte'}
filename_train = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = RW.read_labels(filename_train['labels'])
images = RW.read_image(filename_train['images'])
network = RW.linear_load('mnist_linear.weights')

updating_network = NH.update(network, images[:100], labels[:100])

M.dim(updating_network)
