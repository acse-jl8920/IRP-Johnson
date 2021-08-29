# coding: utf-8
# 2018-04-12 copy of unet.ipynb

import os
import sys
# import tensorflow
from keras.models import Model, load_model
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.layers.merge import Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras import backend as K
from keras import optimizers
from keras import metrics
import datetime
import glob
import random
from pandas import DataFrame
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
class UNet_Experiment(object):
    '''
    UNet_Experiment is a test harness for running a variety of unet modifications quickly
    __init__: 
      version is looking for a string slug to be used in file names
      data_path is the string that will be passed to the glob function to grab files
      targ_path is the string that will be passed to the glob function to grab files
      UNET_ID is set to 0 by default, this indicates which unet will be used for this experiment
      RANDOMIZED is set to True by default, used for various testing
      VERBOSE is set to True by default, used for debugging, also used to decide whether images
        are printed to the screen
    '''
    
    #Recommended import: from unet import UNet_Experiment as ue
    #list of unet options
        
    def __init__(self, test_version, data_path, targ_path, EPOCHS = 0, RANDOMIZED = True, RAND_SEED = 17, VERBOSE = True):
        self.MODEL_NOTSET = -1
        self.MODEL_DEFAULT = 0
        self.MODEL_UNET0 = -99 #choose own parameters
        self.MODEL_UNET1 = 1
        self.MODEL_UNET2 = 2
        self.MODEL_UNET3 = 3
        self.MODEL_UNET4 = 4
        self.MODEL_UNET5 = 5
        self.MODEL_UNET6 = 6
        
        self.version = test_version
        self.randomized = RANDOMIZED
        self.unet_id = -1

        
        self.epochs = EPOCHS
        self.seed = RAND_SEED
        self.verbose = VERBOSE
        self.data_arr = grab_files_list(data_path, self.verbose)
        self.targ_arr = grab_files_list(targ_path, self.verbose)

        self.hist = None
        self.latest_model_fn = '' #TO IMPLEMENT
        
    def setup_data_bins(self, num_train, num_val, num_test=0, px=2048,ops = 0,num_class = 1):
        num_tiles = len(self.data_arr)
        if((num_train + num_val + num_test) > num_tiles):
            print('Data bin distribution exceeds available tiles')
        else:
            tile_order_arr = self.get_tile_order(num_tiles)
            
            # print(tile_order_arr)
            # print(type(tile_order_arr))
            if (len(self.data_arr) == len(self.targ_arr)):
                #make training set
                lst = range(num_train+num_val+num_test)
                
                self.train_dataset = []
                self.train_target = []
                for f in tile_order_arr[0:num_train]:  
                    loadIm(self.data_arr[f], self.targ_arr[f], self.train_dataset, self.train_target, px=px, ops = ops,num_classes = num_class)
                    
                #changes values from [0,1] to [-1,1], convert to numpy array
#                 self.train_dataset = 2*np.array(self.train_dataset)-1 
                #convert to numpy array
                self.train_dataset =  np.asarray(self.train_dataset)

                self.train_target = np.asarray(self.train_target, dtype = np.float16)
                self.train_dataset = self.train_dataset.reshape(self.train_dataset.shape[0], px,px,1)
                self.train_target = self.train_target.reshape(self.train_target.shape[0], px*px,num_class)
                #debugging
                print('Tile #' + str(f))
                print(self.data_arr[f])  
                print(self.targ_arr[f])
                print(self.train_dataset.shape)
                print(self.train_target.shape)
                
                #make validation set
                self.val_dataset = []
                self.val_target = []
                for f in tile_order_arr[num_train:(num_train+num_val)]:    
                    loadIm(self.data_arr[f], self.targ_arr[f], self.val_dataset, self.val_target, px=px,ops = ops, num_classes = num_class)
                                        
                #changes values from [0,1] to [-1,1], convert to numpy array
#                 self.val_dataset = 2*np.array(self.val_dataset)-1 
                self.val_dataset =  np.asarray(self.val_dataset)

                #convert to numpy array
                self.val_target = np.asarray(self.val_target, dtype = np.float16)
                self.val_dataset = self.val_dataset.reshape(self.val_dataset.shape[0], px,px,1)
                self.val_target = self.val_target .reshape(self.val_target.shape[0], px*px,num_class)
                #debugging
                print('Tile #' + str(f))
                print(self.data_arr[f])  
                print(self.targ_arr[f])
                print(self.val_dataset.shape)
                print(self.val_target.shape)

                #TODO: make test set
                self.train_dataset = self.train_dataset/255.
                self.val_dataset = self.val_dataset/255.
#                 del tile_order_arr
#                 del self.data_arr
#                 del self.targ_arr

                

            else: 
                print('Lists of files for the filled and seg images were different sizes. ' + 
                      'No training and test bins created.')


    def set_model(self, UNET_ID):
        self.unet_id = UNET_ID
        if(self.unet_id is self.MODEL_DEFAULT):
            self.model = default_unet()
        elif(self.unet_id is self.MODEL_UNET1):
            self.model = unet1()
        elif(self.unet_id is self.MODEL_UNET2):
            self.model = unet2()
        elif(self.unet_id is self.MODEL_UNET3):
            self.model = unet3()
        elif(self.unet_id is self.MODEL_UNET4):
            self.model = unet4()
        elif(self.unet_id is self.MODEL_UNET5):
            self.model = unet5()
        elif(self.unet_id is self.MODEL_UNET6):
            self.model = unet6()
        elif(self.unet_id is self.MODEL_NOTSET):
            print("!! Model is not set !!")
        else:
            #add more options later
            print("UNET_ID: " + str(self.unet_id))
            self.model = default_unet()


    def compile_test_model(self):
        if (self.model is not None):
            adam = optimizers.Adam(lr=1e-4) #adam = updates, learning
            self.model.compile(adam, loss = 'binary_crossentropy', metrics=['acc']) #loss = log_loss or binary_crossentrophy
            self.model.summary()
        else: 
            print("Model not compiled. No model assigned yet.")


    def train_model(self, addl_ep=500, ex=2, SAVE=True, PLOT=True):
        #epochs_at, dataset, target, testdata, testtarg are all taken from the global variables
        #calculates now for itself
        addl_hist = self.model.fit(self.train_dataset,self.train_target, epochs=addl_ep, validation_data=(self.val_dataset,self.val_target), shuffle=True)
        #history_addl = mod.fit(dataset,target,batch_size = 10, epochs=addl_ep, validation_data=(testdata,testtarg), shuffle=True)

        self.epochs = self.epochs + addl_ep
        print(self.epochs)

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fn_ep = 'models/model_'+ self.version + '_epochs_' + str(self.epochs) + '_' + now + '.h5'
        if(SAVE):
            self.model.save(fn_ep)
        print(fn_ep + ', ' +  str(SAVE))
        
        #mod, ep_at, dataset, target, now, ver, data_type = 'Training', ex=4, SAVE = False
        show_progress(self.model, self.epochs, self.train_dataset, self.train_target, now, self.version, data_type = 'Training', ex = ex, SAVE=True)
        show_progress(self.model, self.epochs, self.val_dataset, self.val_target, now, self.version, data_type = 'Validation', ex = ex, SAVE=True)

        self.add_to_hist(addl_hist, now, PLOT=PLOT)
        return addl_hist
    
    def add_to_hist(self, addl_hist, now=None, PLOT=True):
        #self.hist = self.hist + addl_hist
#        self.hist['loss'] = self.hist['loss'] + addl_hist['loss']
        if(self.hist is None):
            self.hist = addl_hist
        else:
            for x in addl_hist.history.keys():
                self.hist.history[x] = self.hist.history[x] + addl_hist.history[x]
        if(PLOT):
            plot_hist(self.hist, self.version, self.epochs, NOW=now, SAVE=True, SHOW=True)

    def get_tile_order(self, num_tiles):
        rand_arr = list(range(num_tiles))

        if(self.randomized):
            random.seed(self.seed)
            random.shuffle(rand_arr)
        else:
            rand_arr
        return rand_arr
    
    
    def run_experiment(self, BINS = [1,1,0], MODEL_ID=0, init_ep=5, EX=2, SAVE=True, PX=2048):
        num_train, num_val, num_test = BINS
        self.setup_data_bins(num_train,num_val,num_test,px=PX)
        self.set_model(UNET_ID=MODEL_ID)
        self.compile_test_model()
        self.train_model(addl_ep=init_ep, ex=EX, SAVE=SAVE)
    
def plot_hist(hist_addl, VER, EP, NOW, SAVE=True, SHOW=True):
    if (NOW is None):
        NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    #plt.subplot(ex,3,3*i+1)
    #plt.title(data_type)
    #plt.imshow(-dataset[i,:,:,0])
    # plot train and validation loss across multiple runs
    plt.subplot(1,2,1)
    plt.plot(hist_addl.history['loss'], color='blue', label='train')
    plt.plot(hist_addl.history['val_loss'], color='orange', label='validation')
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val (test)'], loc='upper left')
    #plt.show()

    plt.subplot(1,2,2)
    # plot train and validation accuracy across multiple runs
    plt.plot(hist_addl.history['acc'], color='blue', label='train')
    plt.plot(hist_addl.history['val_acc'], color='orange', label='validation')
    plt.title('model train vs validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val (test)'], loc='upper left')

    plt.suptitle('UNet '+ VER + ' at ' + str(EP) + ' Epochs - ' + 'History')
    plt.gcf().set_size_inches((12,5))

    fn = 'plots/' + 'History' + '_model_'+ str(VER) + '_epochs_' + str(EP) +'_' + NOW + '.png'
    print(fn)
    plt.savefig(fn, bbox_inches='tight')
    plt.show()

#        Inputs: string path (regexp) that will cover a list of files or a single file
#                boolean verbose which toggles debugging print statements
#        
#        Return: arr
#        Purpose: pass in list of files that will be iterated over    
def grab_files_list(path, verbose):
    arr = glob.glob(path) #(glob.glob("Robbins_Dataset/synth_out/extracted_04-16km_on_blur2/*.png"))
    arr.sort()
    if (verbose):
        print(len(arr))
        # print(arr)
    return arr
def getLabel(path, n_classes, width, height, resize_op):
    seg_labels = np.zeros((height, width, n_classes))
#     img = cv2.imread(path, 1)
    img = Image.open(path)

    if (resize_op == 1):
        img = img.resize((width, width),Image.NEAREST)
        img = np.asarray(img)
#     elif resize_op == 2:
#         img = cv2_letterbox_image(img, (width, height))
#     img = img[:, :, 0]
    else:
        img = np.asarray(img)
    for c in range(0,n_classes):
        seg_labels[:, :, c] = (img == c).astype(np.float16)
#     seg_labels = np.reshape(seg_labels, (width * height, n_classes))
    del img
    return seg_labels
def loadIm(fname, tname, data, target, num_classes= 4, step=1, newpx = 416, px = 416, ops = 0):    
    im = Image.open(fname)
    #px = 2048 #im.size #TODO: FIX THIS IT WILL BREAK EVERYTHING
    # print('max: ' + str(im.max()) + ', min: ' + str(im.min()) + ', mean: ' + str(im.mean()))
    tim = Image.open(tname) #makes values of target binary
    # counter = 0
    # print(im.shape)
    # print(tim.shape)
    # for y in range(0,px,step): #no need to sub 512 b/c px are mult of 512
    #     for x in range(0,px,step):
    im = im.resize((px, px),Image.NEAREST)
#     tim = tim.resize((px, px),Image.NEAREST)

    tim = getLabel(tname, num_classes, px, px, ops)
    data.append(np.asarray(im))
    target.append(np.asarray(tim))
    del tim
    del im


# combine predDataset and predTestset as one that has a flag for training, validation, or test data set
def show_progress(mod, ep_at, dataset, target, now, ver, data_type = 'Training', ex=4, SAVE = True):    
    #Show progress on training data
    outs = mod.predict(dataset[0:ex]) #call predict on only what's needed
    
    for i in range(ex):
        plt.subplot(ex,3,3*i+1)
        plt.title(data_type)
        plt.imshow(-dataset[i,:,:,0])

        plt.subplot(ex,3,3*i+2)
        plt.title('Target')
        plt.imshow(target[i,:,:,0])

        plt.subplot(ex,3,3*i+3)
        plt.title('Prediction')
        plt.colorbar()
        plt.imshow(outs[i,:,:,0])

    plt.gcf().set_size_inches((12,ex*4))

    plt.suptitle('UNet '+ ver + ' at ' + str(ep_at) + ' Epochs - ' + data_type + ' Data Set')
    if(SAVE):
        fn = 'plots/' + data_type + '_model_'+ str(ver) + '_epochs_' + str(ep_at) +'_' + now + '.png'
        plt.savefig(fn, bbox_inches='tight')
    plt.show()


    
def default_unet(): 
    z1 = Input(shape=(384,384,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(16, 3, padding='same', activation='relu')(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(24, 3, padding='same', activation='relu')(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(32, 3, padding='same', activation='relu')(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(48, 3, padding='same', activation='relu')(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))
    z6 = Conv2D(64, 3, padding='same', activation='relu')(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)
    print('d6: {}'.format(d6.shape))
    print('p6: {}'.format(p6.shape))

    z7 = Conv2D(96, 4, padding='same', activation='relu')(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)
    print('z7: {}'.format(z7.shape))

    z8 = Conv2D(128, 3, padding='same', activation='relu')(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(128, 3, padding='same', activation='relu')(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(128, 3, padding='same', activation='relu')(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])
    print('d9: {}'.format(d9.shape))
    print('q9: {}'.format(q9.shape))
    # print('x8: {}'.format(x8.shape))
    print('d9b: {}'.format(d9b.shape))
   
    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(96, 3, padding='same', activation='relu')(u8)
    d8b = Dropout(0.4)(q8)
    print('a9: {}'.format(a9.shape))
    print('u8: {}'.format(u8.shape))
    print('q8: {}'.format(q8.shape))
    print('d8b: {}'.format(d8b.shape))
    a8 = Add()([d8b,z7])

    
    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(64, 3, padding='same', activation='relu')(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(48, 3, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(32, 3, padding='same', activation='relu')(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(24, 3, padding='same', activation='relu')(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(16, 3, padding='same', activation='relu')(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, 3, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)

def unet1(): 
    ACT = 'relu'
    KERN_SIZE = 3
    FILTERS=[16,24,32,48,64,96,128,128]
    
    z1 = Input(shape=(None,None,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation=ACT)(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(FILTERS[7], KERN_SIZE, padding='same', activation=ACT)(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, KERN_SIZE, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)

def unet2():
    ACT = 'relu'
    KERN_SIZE = 7
    FILTERS=[16,24,32,48,64,96,128,128]
    
    z1 = Input(shape=(None,None,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation=ACT)(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(FILTERS[7], KERN_SIZE, padding='same', activation=ACT)(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, KERN_SIZE, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)


def unet3():
    ACT = 'relu'
    KERN_SIZE = 3
    #FILTERS=[128,128,128,128,128,128,128,128]
    #FILTERS=[16,24,32,48,64,96,128,128]
    #FILTERS = [i / 2 for i in FILTERS]
    FILTERS=[8,12,16,24,32,48,64,64]
    
    z1 = Input(shape=(None,None,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation=ACT)(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(FILTERS[7], KERN_SIZE, padding='same', activation=ACT)(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, KERN_SIZE, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)

def unet4():
    ACT = 'relu'
    KERN_SIZE = 3
    FILTERS=[16,24,32,48,64,96,128,128]
    FILTERS = [i * 2 for i in FILTERS]
    
    
    z1 = Input(shape=(None,None,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation=ACT)(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(FILTERS[7], KERN_SIZE, padding='same', activation=ACT)(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, KERN_SIZE, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)

def unet5():
    ACT = 'relu'
    KERN_SIZE = 11
    FILTERS=[16,24,32,48,64,96,128,128]
    #FILTERS = [i * 2 for i in FILTERS]
    #kern=7 and fil*4 --> resource exhausted
    #kern=3 and fil*4 --> resource exhausted
    #kern=3 and fil*3 --> resource exhausted
    
    z1 = Input(shape=(None,None,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation=ACT)(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(FILTERS[7], KERN_SIZE, padding='same', activation=ACT)(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, KERN_SIZE, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)


def unet6():
    ACT = 'relu'
    KERN_SIZE = 7
    FILTERS=[16,24,32,48,64,96,128,128]
    FILTERS = [i * 2 for i in FILTERS]
    
    z1 = Input(shape=(None,None,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation=ACT)(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(FILTERS[7], KERN_SIZE, padding='same', activation=ACT)(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, KERN_SIZE, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)


def unet0(KERN_SIZE=3, ACT='relu', FILTERS=[16,24,32,48,64,96,128,128]):
    #ACT = 'relu'
    #KERN_SIZE = 3
    #FILTERS=[16,24,32,48,64,96,128,128]
    
    z1 = Input(shape=(None,None,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation=ACT)(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(FILTERS[7], KERN_SIZE, padding='same', activation=ACT)(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(FILTERS[6], KERN_SIZE, padding='same', activation=ACT)(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(FILTERS[5], KERN_SIZE, padding='same', activation=ACT)(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(FILTERS[4], KERN_SIZE, padding='same', activation=ACT)(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(FILTERS[3], KERN_SIZE, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(FILTERS[2], KERN_SIZE, padding='same', activation=ACT)(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(FILTERS[1], KERN_SIZE, padding='same', activation=ACT)(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(FILTERS[0], KERN_SIZE, padding='same', activation=ACT)(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, KERN_SIZE, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)