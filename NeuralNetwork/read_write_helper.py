'''
TASKS: B, C, F, G
'''

import struct as st
import json
import os

### TASK B)

def read_labels(filename):
    '''
    Description:

    Using the struct.unpack function, this function reads the 
    labels from MNIST-data. 

    Assumes that the file is in the same folder as this 
    python document, and the number of labels is 10000.

    Returns list of the labels, and a print message 
    if the magic number is 2049.

    ________

    Arguments: 

    filename = name of the file as a string

    '''
    with open(filename, 'rb') as f:
        f.seek(2) #start after the 2 zeroes
        magic, zeros2, no_items = st.unpack('>HHH',f.read(6)) #magic number as hex digit
        if magic == 2049:
            print(f'The Magic Number is {magic}!')
        else:
            print(f"The magic number is {magic}, which is not 2049.")
        f.seek(8)
        labels = st.unpack(f">{no_items}B", f.read(no_items))
        return list(labels)

### TASK C)

def read_image(filename):
    '''
    Description:

    Using the struct.unpack function, this function reads 
    the image data from the MNIST-database. 
    
    Assumes that the data is stored in the same folder
    as this document.
    
    Returns a list of the image data and print message 
    if the magic number is 2051.

    ________

    Arguments: 

    filename = name of the file as a string
    '''
    with open(filename, 'rb') as f:
        f.seek(2) #start after the 2 zeroes
        magic, zeros2, noIm, zeros3, noR, zeros4, noC = st.unpack('>HHHHHHH', f.read(14)) #magic number as hex digit
        if magic == 2051:
            print(f'The Magic Number is {magic}!')
        else:
            print(f"The Magic Number is {magic}, and not 2051")

        images = list()

        for i in range(noIm):
            image = list()
            for j in range(noR):
                row = list(st.unpack(">28B", f.read(28)))
                image.append(row)
            images.append(image)

    return images

### TASK F)

def linear_load(filename):
    '''
    Description:

    Using the json.load function, this function loads 
    a json string - in our case, a linear network. Gives a
    "FileNotFoundError", if there is no file in the directory
    corresponding to the input string.
    
    Assumes that the data is stored in the same folder
    as this document.
    
    Returns the file as a json string.

    ________

    Arguments: 

    filename = name of the file as a string
    '''
    try:
        with open(filename, "r") as f:
            json_string = json.load(f)
            return json_string
    except FileNotFoundError:
        print(f"Cannot find {filename} in the directory. \nPlease check the filename and the pathing to said filename.")

def linear_save(filename, network): ## inspired by https://stackoverflow.com/questions/42718922/how-to-stop-overwriting-a-file?fbclid=IwAR3osjuyuJTJtvP9wqpBsuBQz8WWTlKmOSpgAmMhn5qXETZ6po7m58GHyAA
    '''
    Description:

    Using the json.dumps function, this function saves 
    a variable to the folder working directory. 
    In our case, it specifically saves a linear network as
    a json-string. Using the os.path.isfile function, 
    the function checks if a file already exists in the 
    directory, it asks the user whether to overwrite the 
    file or not. 
    
    Assumes nothing.
    
    Returns nothing.

    ________

    Arguments: 

    filename = name of the file as a string
    network = variable to save 

    '''
    flag = True
    while flag:
        if os.path.isfile(filename):
            response = input(f"There is already a file named {filename}\nOverwrite? (Yes/No)")
            if response.lower() != "yes":
                break
        network = json.dumps(network)
        with open(filename, "w") as f:
            f.write(network)
            flag = False

### TASK G)
def image_to_vector(image): #inspired by https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    '''
    Description:

    Standardize image to be a single list (image vector).

    Assumes that the image is a 2 dimensional list.
    Assumes the values of the image is between [0, 255].

    Returns a list of floats between [0,1]. 

    ________

    Arguments: 

    image = 2 dimensional list

    '''
    return [(item)/(255) for sublist in image for item in sublist]
