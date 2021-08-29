#coding=utf-8
import tensorflow as tf
import keras 
from keras.models import *
from keras.layers import *
import numpy as np
from metrics import metrics
from losses import LOSS_FACTORY
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

class UNet():
    def __init__(self):
        
        self.model_weights_path = ''
        self.model = self.__build_UNet()
        self.height = 416
        self.width = 416
    
    def __build_UNet(self,nClasses = 2, input_height=416, input_width=416):
        """
        UNet - Basic Implementation
        Paper : https://arxiv.org/abs/1505.04597
        """
        inputs = Input(shape=(input_height, input_width, 1))
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        conv1 = conv_block(inputs, n1)
    
        conv2 = MaxPooling2D(strides=2)(conv1)
        conv2 = conv_block(conv2, filters[1])
    
        conv3 = MaxPooling2D(strides=2)(conv2)
        conv3 = conv_block(conv3, filters[2])
    
        conv4 = MaxPooling2D(strides=2)(conv3)
        conv4 = conv_block(conv4, filters[3])
    
        conv5 = MaxPooling2D(strides=2)(conv4)
        conv5 = conv_block(conv5, filters[4])
    
        d5 = up_conv(conv5, filters[3])
        d5 = Add()([conv4, d5])
    
        d4 = up_conv(d5, filters[2])
        d4 = Add()([conv3, d4])
        d4 = conv_block(d4, filters[2])
    
        d3 = up_conv(d4, filters[1])
        d3 = Add()([conv2, d3])
        d3 = conv_block(d3, filters[1])
    
        d2 = up_conv(d3, filters[0])
        d2 = Add()([conv1, d2])
        d2 = conv_block(d2, filters[0])
    
        o = Conv2D(nClasses, (3, 3), padding='same')(d2)
    
        outputHeight = Model(inputs, o).output_shape[1]
        outputWidth = Model(inputs, o).output_shape[2]
    
        out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
        out = Activation('softmax')(out)
    
        model = Model(inputs=inputs, outputs=out)
        model.outputHeight = outputHeight
        model.outputWidth = outputWidth
    
        return model
    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        
    def complie_model(self, optimizer=None, version = '0', loss = 'ce'):
        '''
        

        Parameters
        ----------
        optimizer : object, optional
            The default is None. It require a optimizer such as Adam or SGD.
                    
        version : str, optional
            The version of your model test. The default is '0'.
        loss : Str, optional
            'ce'	Cross Entropy
            'weighted_ce'	Weighted Categorical loss
            'b_focal'	Binary Focal loss
            'c_focal'	Categorical Focal loss
            'dice'	Dice loss	Yes
            'bce_dice'	BCE + Dice loss
            'ce_dice'	CE + Dice loss
            'g_dice'	Generalized Dice loss
            'jaccard'	Jaccard loss
            'bce_jaccard'	BCE + Jaccard loss
            'ce_jaccard'	CE + Jaccard loss
            'tversky	Tversky' loss
            'f_tversky'	Focal Tversky loss
            The default is 'ce'.

        Returns
        -------
        None.

        '''   
        
    
        csv_logger = CSVLogger(log_file_path, append=False)
        # early_stop = EarlyStopping('loss', min_delta=0.1, patience=patience, verbose=1)

        history = History()
        #set the log save dir, it will save the network value by every epochs in tensorboards.
        tb_cb = keras.callbacks.TensorBoard(log_dir='weights/exp1/'+version+'/log/' , write_images=1, histogram_freq=0)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        self.call_backs = [csv_logger, tb_cb, reduce_lr]
        self.version = version
        if(optimizer == None):
            opt = optimizers.Adam()
        else:
            opt = optimizer
        loss = LOSS_FACTORY[loss]
        self.model.compile(opt, loss =loss, metrics=['accuracy', 'iou_score','f1_score'])   
        
    def train(self, X_train, y_train, X_val, y_val,epochs=20, 
              batch_sizes = 6, weight_pth='weights/exp1/'):
        '''
        

        Parameters
        ----------
        X_train : TYPE
            The array of the training images shape [n,416, 416, 1].
        y_train : TYPE
            The ground turth mask of the training images with shape [n, 416 * 416, 2]
        X_val : TYPE
            validation image
        y_val : TYPE
            validation mask.
        epochs : TYPE, optional
            training epochs. The default is 20.
        batch_sizes : TYPE, optional
            the training batch size. The default is 6.
        weight_pth : TYPE, optional
            the folder to save the trained network weights. The default is 'weights/exp1/'.

        Returns
        -------
        None.

        '''
        hist = self.model.fit(X_train,y_train,batch_size = batch_sizes,
                              callbacks = self.call_backs,epochs=epochs,
                              validation_data=(X_val,y_val), shuffle=True)
        self.model.save_weights(weight_pth+self.version+'.h5')
    def test(self, img, ground_turth):
        '''
        ground_turth: array of mask(shape[num_imgs, height * width, channel(2)] 
        
        '''
        loss = LOSS_FACTORY['ce']
        adam = optimizers.Adam()
        self.model.compile(adam, loss =loss, metrics=['accuracy', 'iou_score','f1_score'])
        if(len(ground_turth.shape)>4):
            shape = ground_turth.shape
            ground_turth.reshape(shape[0], self.width*self.height,2)
        self.model.evaluate(img, ground_turth)
        
        
        
    def detect_mult_img(self, imgs):
        '''
        

        Parameters
        ----------
        imgs : array
            Batch of image with shape [num_img, width, weight]
            for the model in this project is (n,416,416)
        Returns
        -------
        r1 : arrays
            mask of each images, with shape (n, 416, 416)

        '''
        imgs = np.asarray(imgs)
        result = self.model.predict(imgs)
        result = result.reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2],2)
        r1 = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2]))
        r1[result[:,:,:,0]<result[:,:,:,1]] = 1
        return r1
    
    def detect_single_img(self,img, model):
        '''
        detect single image

        Parameters
        ----------
        imgs : array
            Batch of image with shape [num_img, width, weight]
            for the model in this project is (n,416,416)
        Returns
        -------
        r1 : arrays
            mask of each images, with shape (n, 416, 416)

        '''        
        img = np.asarray(img)
        result = self.model.predict(img)
        result = result.reshape(img.shape[0],img.shape[1],2)
        r1 = np.zeros((img.shape[0],img.shape[1]))
        r1[result[:,:,0]<result[:,:,1]] = 1
        return r1