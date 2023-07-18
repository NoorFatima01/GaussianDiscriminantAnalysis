import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype) #x.shape(0) >> No. of rows x.shape(1)+1 >> No. of columns
    new_x[:, 0] = 1 #The first column of new_x is set to 1 using new_x[:, 0] = 1. This column represents the intercept, and all its elements are set to 1.
    new_x[:, 1:] = x #The remaining columns of new_x (starting from the second column) are assigned the values of the corresponding columns from x. The 1: indexing notation means that the assignment should start from the second column of new_x and go up to the last column, while the values are taken from the columns of x.

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
        #headers is a list


    # Load features and labels
    #It identifies the indices of columns starting with 'x' and stores them in x_cols. Similarly, it identifies the indices of the label_col column and stores them in l_cols.
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]

    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)
    #inputs and labels are arrays

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)
    #This if statement checks if the inputs array has a dimensional`ity of 1. If it does, it means that the data has been loaded as a 1-dimensional array (a single column). To ensure compatibility with later calculations, the np.expand_dims() function is used to add an extra dimension to the inputs array. The -1 argument indicates that the new dimension should be added at the last position. i.e. if the previous shape is (n,) The new shape will be (n,1)


    if add_intercept:
        inputs = add_intercept_fn(inputs)
    #Adds intercepts

    return inputs, labels


def plot(x, y, theta, save_path=None, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)
    # x[y == 1, -2] selects the values from the second-to-last column (-2) of the x array where the corresponding elements in the y array are equal to 1.

    # Plot decision boundary (found by solving for theta^T x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path is not None:
        plt.savefig(save_path)