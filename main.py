import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gzip

def load_mnist(path,kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),dtype=np.uint8,offset=16).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_mnist('data', kind='train')
X_test, y_test = load_mnist('data', kind='t10k')

print(X_train[0])

# lookup = {
#     0: 'T-shirt',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
# }

# print(trLabels[10])
# plt.imshow(trImages[10],cmap='gray',vmin=0,vmax=255)
# plt.show()
