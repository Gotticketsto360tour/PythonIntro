'''
TASKS: D, N, O
'''
import matplotlib.pyplot as plt
import math_helper as M
import math
### TASK D)

def plot_images(images, labels, index_list = 10, columns = 5):

    '''
    Description:
    Returns multiple sub-plots of images with labels.
    It uses the subplots and the imshow function from matplotlib.pyplot.
    Uses a "binary" colormap for a clear black and white
    representation. Axis ticks and values are removed for
    aesthetic purposes.

    ________

    Arguments:
    images = list with any number of pixel images (usually 28x28).

    labels = list with any number of labels (e.g., '7') corresponding to images.

    index_list = list containing indexes of which images/labels to plot.
    This can also be specified as an integer, in which case the function
    will simply plot that amount of images, starting from the first
    image in the provided list of images/labels.

    columns = integer specifying how many columns the subplots that
    the function returns are arranged in.

    ________

    Examples:
    for examples, see test.py.

    '''

    if isinstance(index_list, list):
        total_img = len(index_list)
    else:
        total_img = index_list

    rows = math.ceil(total_img/columns)
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        cols_left = min(total_img, columns)
        if total_img < columns:
            for k in range(total_img,columns):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
            if isinstance(index_list, list):
                axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "binary")
                axs[i,j].set_title(labels[index_list[(i*columns)+j]])
            else:
                axs[i,j].imshow(images[(i*columns)+j], cmap = "binary")
                axs[i,j].set_title(labels[(i*columns)+j])
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
        total_img -= columns
    fig.tight_layout()
    return fig

### TASK N)

def plot_images_new(images, labels, index_list = 10, columns = 5, predictions = None):

    '''
    Description:
    Returns multiple sub-plots of images with labels.
    It uses the subplots and the imshow function from matplotlib.pyplot.
    Uses a "binary" colormap for a clear black and white
    representation. Axis ticks and values are removed for
    aesthetic purposes. In addition will display a message
    whenever predictions differ from the correct labels.

    ________

    Arguments:
    images = list with any number of pixel images (usually 28x28).

    labels = list with any number of labels (e.g., '7') corresponding to images.

    index_list = list containing indexes of which images/labels to plot.
    This can also be specified as an integer, in which case the function
    will simply plot that amount of images, starting from the first
    image in the provided list of images/labels.

    columns = integer specifying how many columns the subplots that
    the function returns are arranged in.

    predictions = a list of predictions (in this case concretely
    predictions of digits by a neural network). The format should
    mathc that of labels.

    ________

    Examples:
    for examples, see test.py.

    '''

    if isinstance(index_list, list):
        total_img = len(index_list)
    else:
        total_img = index_list
        index_list = [x for x in range(index_list)]

    if predictions == None:
        predictions = labels

    rows = math.ceil(total_img/columns)
    fig, axs = plt.subplots(rows, columns)
    
    for i in range(rows):
        cols_left = min(total_img, columns)
        if total_img < columns:
            for k in range(total_img, columns):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
            axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "binary")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            if labels[index_list[(i*columns)+j]] == predictions[(i*columns)+j]:
                axs[i,j].set_title(predictions[(i*columns)+j])
            else:
                axs[i,j].imshow(images[index_list[(i*columns)+j]], cmap = "Reds")
                axs[i,j].set_title(f'{predictions[(i*columns)+j]}, correct {labels[index_list[(i*columns)+j]]}', color = 'red')
        total_img -= columns
    fig.tight_layout()
    return fig

### TASK O)

def weights_plot(A, plt_col = 5, image_dim = 28): #weights count = integer.

    '''
    Description:
    Returns multiple sub-plots of heatmaps of the weights
    of a neural network. It uses the subplots and the imshow function
    from matplotlib.pyplot. Axis ticks and values are removed for
    aesthetic purposes. In addition will display a message
    whenever predictions differ from the correct labels.

    ________

    Arguments:
    A = A matrix (list of lists) of weights from a neural network.

    plt_col = Integer specifying how many columns the subplots that
    the function returns are arranged in.

    image_dim = Dimension of the picture to plot. In our case,
    this will always be 28x28, but should generalize to other
    formats. 

    ________

    Examples:
    for examples, see test.py.

    '''

    cols_A = M.gen_col(A)
    rows, columns = M.dim(A)

    # creating K which holds lists of 28x28.
    K = [[] for i in range(columns)]
    for i in range(columns):
        C = [[] for i in range(image_dim)]
        for j in range(image_dim):
            for k in range(image_dim):
                C[j].append(next(cols_A))
        K[i].append(C)

    K = [y for x in K for y in x] #flatten the list.
    #needed for the plot:
    plt_row = math.ceil(columns/plt_col)
    fig, axs = plt.subplots(plt_row, plt_col)

    #plotting
    for i in range(plt_row):
        cols_left = min(columns, plt_col)
        if columns < plt_col:
            for k in range(columns, plt_col):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
            axs[i,j].imshow (K[(i*plt_col)+j], cmap = "gist_heat")
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].set_title((i*plt_col)+j)
        columns -= plt_col
    fig.tight_layout()
    plt.show()
    return fig
