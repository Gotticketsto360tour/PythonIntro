'''
TASKS: H, I, J, K, L, M
'''
import read_write_helper as RW

### HELPER FUNCTIONS

## Dimensions

def dim_recursive(S): #(heavily) inspired by https://stackoverflow.com/questions/17531796/find-the-dimensions-of-a-multidimensional-python-array?fbclid=IwAR1FMXJTEZjeMl08sojxZ4y9wNxCpqZL0EzWPUOSLgvV1YC5jchGvO9Js6I
    '''
    Description: 
    
    Recursive function to calculate the dimensions of 
    a list. 

    Assumes that the input is a list, and that the sublists
    are of equal lengths. 

    Returns a list containing the dimensions of S. 

    ________

    Arguments:

    S = list
    ________  

    Examples:

    >>> dim_recursive([[1,2,3], [2,3,4]])
    [2, 3]

    '''
    if not type(S) == list:
        return []
    return [len(S)] + dim_recursive(S[0])

def dim(S):
    '''
    Description:

    Using the dim_recursive function, this function ensures
    that if a list has only one dimension (row vector), the
    returned list of dimensions will be [1, len(S)]. In all
    other cases, it is the same as dim_recursive.

    Assumes that the input is a list.

    Returns a list containing the dimensions of S. 
    ________

    Arguments:

    S = list 
    ________

    Examples:

    >>> dim([[1, 2, 3], [2, 3, 4]])
    [2, 3]

    >>> dim([1, 2, 3])
    [1, 3]

    '''
    dim_list = dim_recursive(S)
    if len(dim_list) < 2:
        dim_list.insert(0,1)
    return dim_list

## Generators
def gen_row(S):
    '''
    Description:

    Generator function which iterates through the sublists,
    starting from the first, and exhausting each sublist 
    before moving on to the next. 

    Assumes that the input is list of lists, with 2 dimensions.

    Yield an element of the list.
    
    ________

    Arguments:

    S = list of lists with 2 dimensions
    ________

    Examples:

    >>> gen_rows = gen_row([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

    >>> next(gen_rows)
    1
    >>> next(gen_rows)
    2
    >>> next(gen_rows)
    3

    '''
    for i in S:
        for j in i:
            yield j

def gen_col(S):
    '''
    Description:

    Generator function which iterates through the sublists,
    yielding the first element of each sublist. 
    This process in repeated until all elements of the list 
    of lists have been yielded.

    Assumes that the input is list of lists, with 2 dimensions.

    Yield an element of the list.
    
    ________

    Arguments:
    
    S = list
    ________

    Examples:

    >>> gen_columns = gen_col([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

    >>> next(gen_columns)
    1
    >>> next(gen_columns)
    3
    >>> next(gen_columns)
    5

    '''
    for i in range(len(S[0])):
        for j in S:
            yield j[i]

### TASK H)

def add(S, O):
    '''
    Description:

    Adds a matrix to another matrix. Raises a ValueError
    if the matrices do not have the same dimensions.
    In terms of linear algebra, this corresponds
    to the operation S + O, where both S and O are matrices.

    Assumes that both matrices have: 
    (1) the same number of rows and columns 
    (2) 2 dimensions when understood as lists

    Returns a new list of lists, corresponding to the result
    of the addition of the two matrices.
    ________

    Arguments:

    S = list of lists with 2 dimensions
    O = list of lists with 2 dimensions
    ________

    Examples:

    >>> add([[1, 2, 3], [3, 4, 5], [5, 6, 7]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [[2, 4, 6], [7, 9, 11], [12, 14, 16]]

    '''
    if dim(S) != dim(O):
        raise ValueError("The two matrices do not have the same dimensions.")
    rows, columns = dim(S)
    A = gen_row(S)
    B = gen_row(O)
    C = [[] for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            C[i].append(next(A) + next(B))
    return C

def sub(S, O):
    '''
    Description:

    Subtracts a matrix from another matrix. Raises a ValueError
    if the matrices do not have the same dimensions.
    In terms of linear algebra, this corresponds to the
    operation S - O, where both S and O are matrices.

    Assumes that both matrices have: 
    (1) the same number of rows and columns 
    (2) 2 dimensions when understood as lists

    Returns a new list of lists, corresponding to the result of 
    the subtraction of the two matrices.

    ________

    Arguments:

    S = list of lists with 2 dimensions
    O = list of lists with 2 dimensions
    ________

    Examples:

    >>> sub([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [3, 4, 5], [5,6,7]])
    [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    '''
    if dim(S) != dim(O):
        raise ValueError("The two matrices do not have the same dimensions.")
    rows, columns = dim(S)
    A = gen_row(S)
    B = gen_row(O)
    C = [[] for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            C[i].append(next(A) - next(B))
    return C

def scalar_multiplication(S, scalar):
    '''
    Description:

    Multiplies the matrix with a scalar. Raises a ValueError
    if scalar is not either an integer or a float.
    In terms of linear algebra, this corresponds to
    multipliying S * scalar, where S is a matrix and scalar
    is a number.

    Assumes that S is a list of lists with 2 dimensions, and 
    that scalar is either an integer or a float.

    Returns a new list of lists, where S is scaled by 
    a scalar. 
    ________

    Arguments:

    S = list of lists with 2 dimensions
    scalar = integer / float
    ________

    Examples:

    >>> scalar_multiplication([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)
    [[2, 4, 6], [8, 10, 12], [14, 16, 18]]

    '''
    if isinstance(scalar, int) or isinstance(scalar, float):
        generator_self = gen_row(S)
        columns, rows = dim(S)
        C = [[] for i in range(columns)]
        for i in range(columns):
            for j in range(rows):
                C[i].append(next(generator_self)*scalar)
        return C
    else:
        raise ValueError("Scalar must be either of type integer or type float.")

def multiply(S, O):
    '''
    Description:

    Multiplies the matrix with another matrix. 
    Raises a ValueError if the number of columns in S is not 
    the same as the number of rows in O.
    In terms of linear algebra, this corresponds to
    the matrix product of S * O, where S and O are both matrices.
    
    Assumes that S and O are lists of lists with 2 dimensions, 
    and that the number of columns in S is the same as 
    the number of rows in O.
    
    Returns a list of lists, corresponding to matrix product of 
    S * O. 

    ________

    Arguments:

    S = list of lists with 2 dimensions
    O = list of lists with 2 dimensions
    ________

    Examples:

    >>> multiply([[1, 2, 3], [4, 5, 6]], [[2, 3], [4, 5], [6, 7]])
    [[28, 34], [64, 79]]

    >>> multiply([[2, 3], [4, 5], [6, 7]], [[1, 2, 3], [4, 5, 6]])
    [[14, 19, 24], [24, 33, 42], [34, 47, 60]]

    '''
    self_rows, self_columns = dim(S)
    other_rows, other_columns = dim(O)

    if self_columns != other_rows:
        raise ValueError('''The two matrices do not match for matrix multiplication.
    There must be the same number of rows in the first matrix as the number of columns in the second.''')

    C = [[] for i in range(self_rows)]

    sum_of_matrices = 0

    for m in range(self_rows):
        B_cols = gen_col(O)
        for i in range(other_columns):
            for j in range(self_columns):
                sum_of_matrices +=  S[m][j] * next(B_cols)
            C[m].append(sum_of_matrices)
            sum_of_matrices = 0
    return C

def transpose(S):
    '''
    Description:

    Transposes the matrix-equivalent of the input list.
    In terms of linear algebra, this corresponds to
    A^T, where A is a matrix.

    Assumes that S is a list of lists of 2 dimensions.

    Returns a list of lists, where indexes are reversed i.e.
    if the input is S[r][c], the returned list is S[c][r].

    ________

    Arguments:

    S = list of lists with 2 dimensions
    ________

    Examples:

    >>> transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]

    '''
    A = gen_row(S)
    rows, columns = dim(S) 
    C = [[] for i in range(columns)] 
    for i in range(rows): 
        for j in range(columns):
            C[j].append(next(A))
    return(C)

### TASK I)

#also assert that both should be of equal length
def mean_square_error(U, V):
    '''
    Description:

    Calculates the mean square error between two lists.
    Raises a TypeError if either input is not a list, or
    if the two lists do not have the same length. 
    
    Assumes that U and V are both lists of one dimension.
    
    Returns a single number, which is the average of the 
    squared sum of the element-wise difference between the 
    two lists. 

    ________

    Arguments:
    
    U = list of 1 dimension
    V = list of 1 dimension
    ________

    Examples:

    >>> mean_square_error([1,2,3,4], [3,1,3,2])
    2.25

    '''
    if not isinstance(U, list) or not isinstance(V, list):
        raise TypeError("Input must be lists.")
    if len(V) != len(U):
        raise TypeError("The two lists must have the same length.")
    vector_sum = 0
    for i in range(len(U)):
        vector_sum += (V[i]-U[i])**2
    return vector_sum/len(U)

### TASK J)

def argmax(V): ### inspired by https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
    '''
    Description:

    Finds the index of the maximum of a list. Raises a TypeError
    if the input is not a list.

    Assumes that V is a list of a single dimension.

    Returns an integer, corresponding to the index of the
    maximum of a list.

    ________

    Arguments:

    V = list of a single dimension
    ________

    Examples:

    >>> argmax([6, 2, 7, 10, 5])
    3

    '''
    if not isinstance(V, list):
        raise TypeError("Input must be a list.")

    return V.index(max(V))

### TASK K):
def categorical(label, classes = 10):
    '''
    Description:

    Creates a list of length classes, containing zeroes in
    all indexes except for the index "label".

    In our case, label corresponds to the correct
    classification of the image.

    Default value of classes are set to 10, as
    we are only trying to recognize 10 numbers (0-9).

    Assumes that label is an integer, where label < classes.

    Returns a list L of length classes, consisting of zeroes 
    except for L[label], which has the value 1.

    ________

    Arguments:

    label = integer
    classes = integer (Optional)
    ________

    Examples:

    >>> categorical(3)
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    '''
    return [0 if x != label else 1 for x in range(classes)]

### TASK L):
def predict(network, image):
    '''
    Description:
    Multiplies an image vector with the weights of a given 
    network, and adds this product with the bias of the network.
    This corresponds to the networks prediction of what 
    the image is.

    Assumes that network is a nested list, consisting
    of two lists of list. The first containing the weights of 
    the network, and the second containing the bias of 
    the network.

    Returns a list of length equal to b (bias vector).

    ________

    Arguments:
    image = image vector (list) of one dimension.
    network = list of lists two sublists. The first being 
    a list of lists containing the weights of the network. The 
    containing the bias of the network. 

    ________

    Examples:
    >>> predict([[[2,3],[2,2],[1,2],[1,2]],[2,3]], [1,2,4,0])
    [12, 18]

    '''
    A, b = network
    image = [image] #manual for now
    xA = multiply(image, A)
    b = [b] #manual for now.
    xAb = add(xA, b)
    xAb_unlisted = xAb[0]
    return xAb_unlisted

### TASK M)

def evaluate(network, images, labels):
    '''
    Description:

    Evaluates the predictions made by a network on list
    of images and their corresponding labels. 
    This evaluation is made on the basis of using 
    mean square error as a cost function. 
    The function calculates the mean cost as well as the 
    mean accuracy. Mean accuracy is given in percent.

    Assumes that the images and labels correspond i.e.
    image[i] has label labels[i]. 

    Returns a tuple (predictions, cost, accuracy), where 
    predictions is a list of all the predictions made,
    cost is a float representing the mean cost of predictions
    and accuracy is the percentage of correct predictions.

    ________

    Arguments: 
    image = list of images
    network = list of lists two sublists. The first being 
    a list of lists containing the weights of the network. The 
    containing the bias of the network. 
    labels = list of labels


    '''
    predictions = []
    cost = 0
    accuracy = 0
    for i in range(len(images)):
        image_vector = RW.image_to_vector(images[i])
        prediction = predict(network, image_vector)
        prediction_label = argmax(prediction)
        cost += mean_square_error(prediction, categorical(labels[i]))
        if prediction_label == labels[i]:
            accuracy += 1
        predictions.append(prediction_label)
    return (predictions, cost/len(images), 100 * accuracy/len(images))

### OPTIONAL TASKS
### ______________

### TASK S) 

### NOTE: TASK S was not completed fully. We left the steps we took to solve the task, as we hope to come back to it after the handin.

import math

def softmaxx(V):
    summing = sum([math.exp(x) for x in V])
    return [math.exp(x)/summing for x in V]

def CE(V, U):
    V_mark = softmaxx(V)
    return -sum([U[i] * math.log(V_mark[i]) for i in range(len(U))])

def evaluate_CE(network, images, labels):  
    predictions = []
    cost = 0
    accuracy = 0
    for i in range(len(images)):
        image_vector = RW.image_to_vector(images[i])
        prediction = predict(network, image_vector)
        prediction_label = argmax(prediction)
        cost += CE(prediction, categorical(labels[i]))
        if prediction_label == labels[i]:
            accuracy += 1
        predictions.append(prediction_label)
    return (predictions, cost/len(images), accuracy/len(images))

import doctest
doctest.testmod(verbose=True)
