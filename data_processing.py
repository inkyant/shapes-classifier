import imageio
import numpy as np
import os
from pathlib import Path
from six.moves import cPickle as pickle
import ntpath

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

## DATASET LINK: https://github.com/frobertpixto/hand-drawn-shapes-dataset


seed1 = 122
np.random.seed(seed1)

BASEDIR = "."

DATA_DIR       = os.path.join(BASEDIR, "data")

PICKLE_DIR     = os.path.join(BASEDIR, "pickles")
DATAFILE = os.path.join(PICKLE_DIR, 'data.pickle')

# Set image properties
image_size  = 70 # Pixel width and height
pixel_depth = 255.0  # Number of levels per pixel


output_labels = {
  'other':     0,
  'ellipse':   1,
  'rectangle': 2,
  'triangle':  3
}

def get_label_for_shape(shape_dir):
    shape = os.path.basename(shape_dir)
    if shape not in output_labels.keys():
        raise Exception('Unknown shape: %s' % shape)
    else:
        return output_labels[shape]
    

# Functions for getting array of directory paths and array of file paths
def get_dir_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n))]

def get_file_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, n))]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


# Normalize image by pixel depth by making it white on black instead of black on white
def normalize_image(image_file, pixel_depth):
    try:
        array = imageio.imread(image_file)
    except ValueError:
        raise

    return 1.0 - (array.astype(float))/pixel_depth  # (1 - x) will make it white on black


def save_to_pickle(pickle_file, object, force=True):
    """
    Save an object to a pickle file
    """       
    if os.path.exists(pickle_file) and not force:
        print(f'{pickle_file} already present, skipping pickling')
    else:
        try:
            with open(pickle_file, 'wb') as file:
                pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'Unable to save object to {pickle_file}: {e}')
            raise


def load_images_for_shape(shape_directory, pixel_depth, user_images,
                          user_images_label, label, verbose=False, min_nimages=1):
    """
    Load all images for a specific user and shape
    """      

    if verbose:
        print("directory for load_images_for_shapes: ", shape_directory)

    image_files = get_file_paths(shape_directory)
    image_index = 0

    for image_file in image_files:
        try:
            if path_leaf(image_file).startswith('.'):  # skip files like .DSStore
                continue

            image_data_all_channels = normalize_image(image_file, pixel_depth)
            image_data = image_data_all_channels[:, :, 0]

            user_images.append(image_data)
            user_images_label.append(label)
            image_index += 1
        except Exception as error:
            print(error)
            print('Skipping because of not being able to read: ', image_file)

    if image_index < min_nimages:
        raise Exception('Fewer images than expected: %d < %d' % (image_index, min_nimages))
    

def load_images_for_user(user_directory, pixel_depth,
                         user_images, user_images_label,
                         verbose=False):
    """
    Load all images for a specific user
    """      
    
    images_dir = os.path.join(user_directory, "images")

    if verbose:
        print("directory for load_images_for_shapes: ", images_dir)

    shape_dirs = get_dir_paths(images_dir)
    for dir in shape_dirs:
        label = get_label_for_shape(dir)
        if label >= 0:
            load_images_for_shape(dir, pixel_depth, user_images, user_images_label, label)

def plot_sample(image, axs):
    axs.imshow(image, cmap="gray")

def display_images(X, Y, alt_title=None):
    """ 
    This function shows images with their real labels
    Presentation is rows of 10 images
    """

    fig = plt.figure(figsize=(13, 10))
    fig.subplots_adjust(hspace=0.2,wspace=0.2,
                        left=0, right=1, bottom=0, top=1.7)
    nb_pictures = len(X)
    nb_per_row = 10
    nb_of_row  = (nb_pictures - 1) // nb_per_row + 1

    for i in range(nb_pictures):
        ax = fig.add_subplot(nb_of_row, nb_per_row, i+1, xticks=[], yticks=[]) 
        plot_sample(X[i], ax)
        if alt_title:
            ax.set_title(alt_title)
        else:
            ax.set_title("{}".format(list(output_labels.keys())[Y[i]]))
    plt.show()


if __name__ == '__main__':

    # Create Pickle directory
    Path(PICKLE_DIR).mkdir(parents=True, exist_ok=True)


    # Get directory and file paths of Shape data
    data_paths = get_dir_paths(DATA_DIR)
    print(f"Dataset contains 1 directory per user:")
    print(data_paths)

    all_images = []
    all_images_label = []

    for user_dir in data_paths:
        load_images_for_user(user_dir, pixel_depth, all_images, all_images_label)

    data = np.array(all_images)
    labels = np.array(all_images_label)

    print('data shape: ', data.shape)
    print('labels shape: ', labels.shape)

    unique, counts = np.unique(labels, return_counts=True)
    print("train label dist.: ", dict(zip(unique, counts)))

    # Save data to single pickle file
    save_to_pickle(
        DATAFILE,
        {
            'data': data,
            'labels': labels
        }
    )

    print('Data Set Pickle saved in: ', DATAFILE)

    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)
    first_random_indexes = indexes[:100]

    display_images(data[first_random_indexes],labels[first_random_indexes])
