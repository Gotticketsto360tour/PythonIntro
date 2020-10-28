'''
TASKS: P, Q, R, and parts of S
'''

import math
import random
import math_helper as M
import read_write_helper as RW

### TASK P)

def create_batches(values, batch_size):
    '''
    Description:

    Using the random.shuffle function from the random module,
    this function partitions a list of values into random batches
    of length "batch_size".
    The only exception is the last batch, which can be of
    a smaller length. If "batch_size" is not an integer,
    a TypeError exception is raised.

    Assumes that the input is a list and
    that batch_size is an integer.

    Returns a list of batches of values.
    ________

    Arguments:

    values = list
    batch_size = integer
    ________
    '''
    if isinstance(batch_size, int):
        values_list = []
        values_copy = values[:]
        random.shuffle(values_copy)
        current_batch = 0

        while current_batch < len(values_copy):
            current_batch += batch_size
            values_list.append(values_copy[current_batch-batch_size:current_batch])
    else:
        raise TypeError("batch_size must be an integer.")

    return values_list

### TASK Q)

def update(network, images, labels, sigma = 0.1):
    '''
    Description:

    Updates the weights and bias of a network. Using the
    mean square error as a cost function, this function
    calculates one step of gradient descent, where
    the stepsize is given by sigma.

    Uses the modules from the math_helper module and thus
    inherits their assumptions and limitations. If need be,
    check their documentation.

    Returns the updated network.
    ________

    Arguments:

    network = list of lists two sublists. The first being
    a list of lists containing the weights of the network. The
    containing the bias of the network.
    images = list of images
    labels = list of labels
    sigma = float (Optional)
    ________
    '''
    A, b = network
    A_list = [[0]*len(network[1]) for i in range(len(A))]
    b_list = [[0 for i in range(len(b))]]

    for n in range(len(images)):
        x = RW.image_to_vector(images[n])
        a = M.predict(network, x)
        y = M.categorical(labels[n])

        for j in range(len(b)):

            current_element = 2 * (a[j] - y[j]) / 10

            b_list[0][j] += current_element

            for i in range(len(A)):
                A_list[i][j] += x[i] * current_element

    b_list_final = M.scalar_multiplication(b_list, (sigma * 1/len(images)))
    b = M.sub([b], b_list_final)

    A_list_final = M.scalar_multiplication(A_list, (sigma * 1/len(images)))
    A = M.sub(A, A_list_final)

    return [A, b[0]]

### TASK R)

def learn(images, labels, epochs, batch_size):
    '''
    Description:

    Initializes a network consisting of random weights and
    biases. The network is then trained using the "update"
    function over a batch of images and labels.
    For each epoch, the images are partioned
    into smaller batches of images and labels.
    The network is succesively updated for each batch.

    Furthermore, the function prints what
    epoch and batch number it is currently training on.
    This is followed by another print, which is the
    accuracy of the updated network and the accuracy
    of the previous network.

    Returns the best performing network in terms of accuracy
    ________

    Arguments:

    images = list of images
    labels = list of labels
    epochs = integer
    batch_size = integer
    ________
    '''

    #initializing the random network:
    b = [random.uniform(0, 1) for m in range(10)]
    A = [[random.uniform(0, 1/784) for n in range(10)] for n in range(784)]
    network = [A, b]
    prev_acc = 0

    print("******** STARTING TO LEARN ********")

    for e in range(epochs):
        batch_number = 0
        batches = create_batches(list(range(len(images))), batch_size)

        for i in batches:
            batch_number += 1
            one_img_batch = [images[j] for j in i]
            one_lab_batch = [labels[j] for j in i]
            print(f"Current Epoch: {e+1} | Current batch: {batch_number}\n_____________________________________")
            network = update(network, one_img_batch, one_lab_batch, sigma = 0.1)
            pred, cost, acc = M.evaluate(network, images, labels)
            if prev_acc <= acc:
                print(f"\nNew record of accuracy achieved!\nCurrent Accuracy: {acc:.2f}\nPrevious Accuracy: {prev_acc:.2f}\n")
                prev_acc = acc
                best_network = network
            else:
                print(f"\nAccuracy did not improve in this batch.\nCurrent Accuracy: {acc:.2f}\nPrevious Accuracy: {prev_acc:.2f}\n")
    print("******** FINISHED LEARNING ********")
    return best_network

def fast_learn(images, labels, epochs, batch_size):
    '''
    Description:

    Initializes a network consisting of random weights and
    biases. The network is then trained using the "update"
    function over a batch of images and labels.
    For each epoch, the images are partioned
    into smaller batches of images and labels.
    The network is succesively updated for each batch.

    Furthermore, the function prints what
    epoch and batch number it is currently training on.
    This is followed by another print, which is the
    accuracy of the updated network and the accuracy
    of the previous network.

    Returns the best performing network in terms of accuracy
    ________

    Arguments:

    images = list of images
    labels = list of labels
    epochs = integer
    batch_size = integer
    ________
    '''
    #initializing the random network:
    b = [random.uniform(0, 1) for m in range(10)]
    A = [[random.uniform(0, 1/784) for n in range(10)] for n in range(784)]
    network = [A, b]
    prev_acc = 0

    print("******** STARTING TO LEARN ********")

    for e in range(epochs):
        batch_number = 0
        batches = create_batches(list(range(len(images))), batch_size)

        for i in batches:
            batch_number += 1
            one_img_batch = [images[j] for j in i]
            one_lab_batch = [labels[j] for j in i]
            print(f"Current Epoch: {e+1} | Current batch: {batch_number}\n_____________________________________")
            network = update(network, one_img_batch, one_lab_batch, sigma = 0.1)
    print("******** FINISHED LEARNING ********")
    return network

### OPTIONAL TASKS:
### ______________

### TASK S

import math 

### NOTE: The following functions DO NOT currently work!! Most probably due to some error in implementing the derivative.

def update_CE(network, images, labels, sigma = 0.1):
    A, b = network
    A_list = [[0]*len(network[1]) for i in range(len(A))]
    b_list = [[0 for i in range(len(b))]]

    for n in range(len(images)):
        x = RW.image_to_vector(images[n])
        a = M.predict(network, x)
        y = M.categorical(labels[n])

        for j in range(len(b)):

            current_element = math.exp(a[j]) / (sum([math.exp(x) for x in a]) - y[j])

            b_list[0][j] += current_element

            for i in range(len(A)):
                A_list[i][j] += x[i] * current_element

    b_list_final = M.scalar_multiplication(b_list, (sigma * 1/len(images)))
    b = M.sub([b], b_list_final)

    A_list_final = M.scalar_multiplication(A_list, (sigma * 1/len(images)))
    A = M.sub(A, A_list_final)

    return [A, b[0]]

def learn_CE(images, labels, epochs, batch_size): ## CURRENTLY *DOES NOT WORK* - PERFORMS TERRIBLY!!! Probably due to some misinterpretation in the derivative.
    #initializing the random network:
    b = [random.uniform(0, 1) for m in range(10)]
    A = [[random.uniform(0, 1/784) for n in range(10)] for n in range(784)]
    network = [A, b]
    prev_acc = 0

    print("******** STARTING TO LEARN ********")

    for e in range(epochs):
        batch_number = 0
        batches = create_batches(list(range(len(images))), batch_size)
        for i in batches: #this should be smarter..
            batch_number += 1
            one_img_batch = [images[j] for j in i]
            one_lab_batch = [labels[j] for j in i]
            print(f"Current Epoch: {e+1} | Current batch: {batch_number}\n_____________________________________")
            network = update_CE(network, one_img_batch, one_lab_batch, sigma = 0.1)
            pred, cost, acc = M.evaluate_CE(network, images, labels)
            if prev_acc <= acc:
                print(f"\nNew record of accuracy achieved!\nCurrent Accuracy: {acc:.2f}\nPrevious Accuracy: {prev_acc:.2f}\n")
                prev_acc = acc
                best_network = network
            else:
                print(f"\nAccuracy did not improve in this batch.\nCurrent Accuracy: {acc:.2f}\nPrevious Accuracy: {prev_acc:.2f}\n")
    print("******** FINISHED LEARNING ********")
    return best_network
