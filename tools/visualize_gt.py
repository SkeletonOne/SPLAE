import os
import matplotlib.pyplot as plt

def visualize_gt(DATASET_PATH):
    plt.rcParams['figure.figsize'] = [20, 10]

    row = 3; col = 6

    for image_index in range(1, row * col + 1):
        plt.subplot(row, col, image_index)
        if image_index % 2 == 1:
            image = plt.imread(os.path.join(DATASET_PATH, str(image_index) + '.png'))
        else:
            image = plt.imread(os.path.join(DATASET_PATH, str(image_index) + '_mask.png'))
        if image_index <= col:
            if image_index % 2 == 1:
                plt.title('Prostate MRI Scanns')
            else:
                plt.title('Prostate GT Mask')
        plt.axis('off')
        plt.imshow(image, cmap = 'gray')
    plt.savefig('Visualized_data.png')
