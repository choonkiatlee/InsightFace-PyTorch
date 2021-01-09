import os
import pickle

import cv2 as cv
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm

from config import path_imgidx, path_imgrec, IMG_DIR, pickle_file, img_batch_size
#from utils import ensure_folder

import glob
import numpy as np


# from multiprocessing import Pool
# from typing import List, Dict, Set, Generator, Tuple
# import itertools
# from functools import partial

def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)



if __name__ == "__main__":
    ensure_folder(IMG_DIR)

    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    
    SPLIT_PICKLE_FILES = True
    parted_pickle_file_fmt_string = "data/{0}.pkl"

    samples = []
    class_ids = set()

    ignored_idxs = []

    n_batches = 10000000 // img_batch_size
    save_every_x_batches = n_batches // 10
    try:
        # for i in tqdm(range(10000000)):
        for batch_no in tqdm(range(45487, n_batches)):     # 39061       

<<<<<<< HEAD
            # filename = '{}.jpg'.format(batch_no)
=======
            filename = '{}.jpg'.format(batch_no)
>>>>>>> 199c2ed6fcd9e8d260a018772adf62917b1032c5
            img_list = []
            labels = []

            n_processed = 0
            img_num = 0
            while n_processed < img_batch_size:

                img_num += 1
            # for img_num in range(img_batch_size):

                # Read record
                header, s = recordio.unpack(imgrec.read_idx(batch_no * img_batch_size + img_num + 1))  

                # Process Header
                try:
                    label = int(header.label)
                except Exception as err:
                    # print("Could not convert label, continuing")
                    continue

                labels.append(label)
                class_ids.add(label)

                # Read image
                img = mx.image.imdecode(s).asnumpy()
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

                img_list.append(img)

                n_processed += 1

            # Save image
            img = np.hstack(img_list)
<<<<<<< HEAD
            filename = os.path.join(IMG_DIR, '{}.jpg'.format(batch_no))
=======
            filename = os.path.join(IMG_DIR, filename)
>>>>>>> 199c2ed6fcd9e8d260a018772adf62917b1032c5
            cv.imwrite(filename, img)

            samples.append({'img': filename, 'labels': labels})


            # if not os.path.exists(filename):

            #     # Save image
            #     img = mx.image.imdecode(s).asnumpy()
            #     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            #     filename = os.path.join(IMG_DIR, filename)
            #     cv.imwrite(filename, img)

            if SPLIT_PICKLE_FILES:
                if (batch_no % save_every_x_batches) == 0 and (batch_no > 0):
                    with open(parted_pickle_file_fmt_string.format(batch_no), 'wb') as file:
                        pickle.dump(samples, file)
                        print('num_samples: ' + str(len(samples)))

                    samples = []

        with open(parted_pickle_file_fmt_string.format(batch_no), 'wb') as file:
            pickle.dump(samples, file)
            print('num_samples: ' + str(len(samples)))

    except Exception as err:
        print(err)


    # if SPLIT_PICKLE_FILES:
    #     # Combine all files 
    #     samples = []

    #     for filename in glob.glob('data/*00.pkl'):

    #         print("Processing File {0}".format(filename))
    #         with open(filename, 'rb') as infile:

    #             sample_file = pickle.load(infile)
    #             samples += sample_file
    
    # print(len(samples))

    # with open("combined.pkl", 'wb') as file:
    #     pickle.dump(samples, file)


    print('num_samples: ' + str(len(samples)))

    class_ids = list(class_ids)
    print(len(class_ids))
    print(max(class_ids))


# if __name__ == "__main__":
#     ensure_folder(IMG_DIR)
#     imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    
#     SPLIT_PICKLE_FILES = True
#     parted_pickle_file_fmt_string = "data/{0}.pkl"

#     samples = []
#     class_ids = set()


#     # %% 1 ~ 5179510

#     try:
#         # for i in tqdm(range(10000000)):
#         for i in tqdm(range(500000, 1001000)):
#             # print(i)

#             filename = '{}.jpg'.format(i)

#             # Read record
#             header, s = recordio.unpack(imgrec.read_idx(i + 1))  

#              # Process Header
#             label = int(header.label)
#             class_ids.add(label)

#             samples.append({'img': filename, 'label': label})


#             if not os.path.exists(filename):

#                 # Save image
#                 img = mx.image.imdecode(s).asnumpy()
#                 img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
#                 filename = os.path.join(IMG_DIR, filename)
#                 cv.imwrite(filename, img)

#             if SPLIT_PICKLE_FILES:
#                 if (i % 500000) == 0 and (i > 0):
#                     with open(parted_pickle_file_fmt_string.format(i), 'wb') as file:
#                         pickle.dump(samples, file)
#                         print('num_samples: ' + str(len(samples)))

#                     samples = []

#             # except KeyboardInterrupt:
#             #     raise
#     except Exception as err:
#         print(err)

    
    # if SPLIT_PICKLE_FILES:
    #     # Combine all files 
    #     samples = []

    #     for filename in glob.glob('data/*00.pkl'):

    #         print("Processing File {0}".format(filename))
    #         with open(filename, 'rb') as infile:

    #             sample_file = pickle.load(infile)
    #             samples += sample_file
    
    # print(len(samples))

    # with open("combined.pkl", 'wb') as file:
    #     pickle.dump(samples, file)

    # print('num_samples: ' + str(len(samples)))

    # class_ids = list(class_ids)
    # print(len(class_ids))
    # print(max(class_ids))

    

