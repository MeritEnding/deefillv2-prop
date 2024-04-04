import tensorflow as tf
import numpy as np
from DeepFill_utils import *
from config import Config
import sn

FLAGS = Config('./inpaint.yml')
img_shape = FLAGS.img_shapes
IMG_HEIGHT = img_shape[0]
IMG_WIDTH = img_shape[1]

class gen_conv_block(tf.keras.layers.Layer):
  def __init__(self, filters, size, stride=1, dilation_rate=1, activation=tf.keras.activations.swish):
    super(gen_conv_block, self).__init__(name='')
    self.filters = filters
    self.activation = activation
    self.conv2d = tf.keras.layers.Conv2D(filters, size, stride, padding='same',
                                        dilation_rate=dilation_rate, activation=None)

  def call(self, input_tensor, training=False):
    x = self.conv2d(input_tensor)
    if self.filters == 3 or self.activation is None:
      return x

    x, y = tf.split(x, num_or_size_splits=2, axis=3)

    x = self.activation(x)
    y = tf.keras.activations.sigmoid(y)
    x = x*y 
    #print(f'shape of x: {x}')
    return x

class gen_deconv_block(tf.keras.layers.Layer):
  def __init__(self, filters, multi=0):
    super(gen_deconv_block, self).__init__(name='')

    self.multi = multi
    # ΙΣΩΣ ΜΕ UPSAMPLING;
    #self.up2d = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest')
    self.gen_conv_block = gen_conv_block(filters, 3, 1)

  def call(self, input_tensor, training=False):
    #x = self.up2d(input_tensor)
    x = input_tensor
    if not self.multi:
        x = resize(x, func='nearest') #otan exo to generatormulticolumn den to thelo
    x = self.gen_conv_block(x)
    return x

class dis_block(tf.keras.layers.Layer):
  def __init__(self, filters, size=5, stride=2):
    super(dis_block, self).__init__(name='')
    self.sn_conv = sn.SpectralNormalization(
        tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same', 
                               activation=tf.keras.layers.LeakyReLU(alpha=0.2))
    )

  def call(self, inputs):
    return self.sn_conv(inputs)

#GENERATOR MULTI COLUMN 
class GeneratorMultiColumn(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        cnum = 48

        #브랜치0 : 추가됨
        self.g0_1 = gen_conv_block(cnum, 9, 1)# x.shape= (32, 64, 64, 24)
        self.g0_2 = gen_conv_block(2*cnum, 9, 2)# x.shape= (32, 32, 32, 48)
        self.g0_3 = gen_conv_block(2*cnum, 9, 1)# x.shape= (32, 32, 32, 48)
        self.g0_4 = gen_conv_block(4*cnum, 9, 1)# x.shape= (32, 32, 32, 96)
        self.g0_5 = gen_conv_block(4*cnum, 9, 1)# x.shape= (32, 32, 32, 96)
        self.g0_6 = gen_conv_block(4*cnum, 9, 1)# x.shape= (32, 32, 32, 96)
        self.g0_7 = gen_conv_block(4*cnum, 9, dilation_rate=2)# x.shape= (32, 32, 32, 96)
        self.g0_8 = gen_conv_block(4*cnum, 9, dilation_rate=4)# x.shape= (32, 32, 32, 96)
        self.g0_9 = gen_conv_block(4*cnum, 9, dilation_rate=8)# x.shape= (32, 32, 32, 96)
        self.g0_10 = gen_conv_block(4*cnum, 9, dilation_rate=16)# x.shape= (32, 32, 32, 96)
        self.g0_11 = gen_conv_block(4*cnum, 9, dilation_rate=32)# x.shape= (32, 32, 32, 96)
        self.g0_12 = gen_conv_block(4*cnum, 7, 1)# x.shape= (32, 32, 32, 96)
        self.g0_13 = gen_conv_block(4*cnum, 7, 1)# x.shape= (32, 32, 32, 96) ==> 이후에 resize를 통해 원크기로 일치시킴


        self.g1_1 = gen_conv_block(cnum, 7, 1)# x_w_mask.shape= (32, 64, 64, 5) # x.shape= (32, 64, 64, 24)
        self.g1_2 = gen_conv_block(2*cnum, 7, 2)# x.shape= (32, 32, 32, 48)
        self.g1_3 = gen_conv_block(2*cnum, 7, 1)# x.shape= (32, 32, 32, 48)
        self.g1_4 = gen_conv_block(4*cnum, 7, 2)# x.shape= (32, 16, 16, 96)
        self.g1_5 = gen_conv_block(4*cnum, 7, 1)# x.shape= (32, 16, 16, 96)
        self.g1_6 = gen_conv_block(4*cnum, 7, 1)# x.shape= (32, 16, 16, 96)
        self.g1_7 = gen_conv_block(4*cnum, 7, dilation_rate=2)# x.shape= (32, 16, 16, 96)
        self.g1_8 = gen_conv_block(4*cnum, 7, dilation_rate=4)# x.shape= (32, 16, 16, 96)
        self.g1_9 = gen_conv_block(4*cnum, 7, dilation_rate=8)# x.shape= (32, 16, 16, 96)
        self.g1_10 = gen_conv_block(4*cnum, 7, dilation_rate=16)# x.shape= (32, 16, 16, 96)
        self.g1_11 = gen_conv_block(4*cnum, 7, dilation_rate=32)# x.shape= (32, 16, 16, 96)
        self.g1_12 = gen_conv_block(4*cnum, 7, 1)# x.shape= (32, 16, 16, 96)
        self.g1_13 = gen_conv_block(4*cnum, 7, 1)# x.shape= (32, 16, 16, 96) ==> 이후에 resize를 통해 절반크기로 일치시킴
        self.g1_14 = gen_deconv_block(2 * cnum, multi=1)  # g1_14 x.shape= (32, 32, 32, 48)
        self.g1_15 = gen_conv_block(2 * cnum, 5, 1)  # g1_15 x.shape= (32, 32, 32, 48) ==> 이후에 resize를 통해 원크기로 일치시킴

        self.g2_1 = gen_conv_block(cnum, 5, 1)# g2_1 x.shape= (32, 64, 64, 24)
        self.g2_2 = gen_conv_block(2*cnum, 5, 2)# g2_2 x.shape= (32, 32, 32, 48)
        self.g2_3 = gen_conv_block(2*cnum, 5, 1)# g2_3 x.shape= (32, 32, 32, 48)
        self.g2_4 = gen_conv_block(4*cnum, 5, 2)# g2_4 x.shape= (32, 16, 16, 96)
        self.g2_5 = gen_conv_block(4*cnum, 5, 1)# g2_5 x.shape= (32, 16, 16, 96)
        self.g2_6 = gen_conv_block(4*cnum, 5, 1)# g2_6 x.shape= (32, 16, 16, 96)
        self.g2_7 = gen_conv_block(4*cnum, 5, dilation_rate=2)# g2_7 x.shape= (32, 16, 16, 96)
        self.g2_8 = gen_conv_block(4*cnum, 5, dilation_rate=4)# g2_8 x.shape= (32, 16, 16, 96)
        self.g2_9 = gen_conv_block(4*cnum, 5, dilation_rate=8)# g2_9 x.shape= (32, 16, 16, 96)
        self.g2_10 = gen_conv_block(4*cnum, 5, dilation_rate=16)# g2_10 x.shape= (32, 16, 16, 96)
        self.g2_11 = gen_conv_block(4*cnum, 5, dilation_rate=32)# g2_11 x.shape= (32, 16, 16, 96)
        self.g2_12 = gen_conv_block(4*cnum, 5, 1)# g2_12 x.shape= (32, 16, 16, 96)
        self.g2_13 = gen_conv_block(4*cnum, 5, 1)# g2_13 x.shape= (32, 16, 16, 96) ==> 이후에 resize를 통해 절반크기로 일치시킴
        self.g2_14 = gen_deconv_block(2*cnum, multi=1) # g2_14 x.shape= (32, 32, 32, 48)
        self.g2_15 = gen_conv_block(2*cnum, 5, 1)# g2_15 x.shape= (32, 32, 32, 48) ==> 이후에 resize를 통해 원크기로 일치시킴

        self.g3_1 = gen_conv_block(cnum, 5, 1)# g3_1 x.shape= (32, 32, 32, 24)
        self.g3_2 = gen_conv_block(2*cnum, 3, 2)# g3_2 x.shape= (32, 32, 32, 48)
        self.g3_3 = gen_conv_block(2*cnum, 3, 1)# g3_3 x.shape= (32, 32, 32, 48)
        self.g3_4 = gen_conv_block(4*cnum, 3, 2)# g3_4 x.shape= (32, 16, 16, 96)
        self.g3_5 = gen_conv_block(4*cnum, 3, 1)# g3_5 x.shape= (32, 16, 16, 96)
        self.g3_6 = gen_conv_block(4*cnum, 3, 1)# g3_6 x.shape= (32, 16, 16, 96)
        self.g3_7 = gen_conv_block(4*cnum, 3, dilation_rate=2)# g3_7 x.shape= (32, 16, 16, 96)
        self.g3_8 = gen_conv_block(4*cnum, 3, dilation_rate=4)# g3_8 x.shape= (32, 16, 16, 96)
        self.g3_9 = gen_conv_block(4*cnum, 3, dilation_rate=8)# g3_9 x.shape= (32, 16, 16, 96)
        self.g3_10 = gen_conv_block(4*cnum, 3, dilation_rate=16)# g3_10 x.shape= (32, 16, 16, 96)
        self.g3_11 = gen_conv_block(4*cnum, 3, dilation_rate=32)# g3_11 x.shape= (32, 16, 16, 96)
        self.g3_12 = gen_conv_block(4*cnum, 3)# g3_12 x.shape= (32, 16, 16, 96)
        self.g3_13 = gen_conv_block(4*cnum, 3, 1)# g3_13 x.shape= (32, 16, 16, 96) ==> 이후에 resize를 통해 절반크기로 일치시킴
        self.g3_14 = gen_deconv_block(2*cnum, multi=1)# g3_14 x.shape= (32, 32, 32, 48)
        self.g3_15 = gen_conv_block(2*cnum, 3, 1)# g3_15 x.shape= (32, 32, 32, 48) ==> 이후에 resize를 통해 원크기로 일치시킴
        self.g3_16 = gen_deconv_block(cnum, multi=1)# g3_16 x.shape= (32, 64, 64, 24)
        self.g3_17 = gen_conv_block(cnum//2, 3, 1)#x_b3.shape= (32, 64, 64, 12)

        self.m1 = gen_conv_block(cnum//2, 3, 1)        
        self.m2 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation=None)        

        # conv_branch1
        self.g18 = gen_conv_block(cnum, 5, 1) #x.shape= (32, 64, 64, 24)
        self.g19 = gen_conv_block(cnum, 3, 2) #x.shape= (32, 32, 32, 24)
        self.g20 = gen_conv_block(2*cnum, 3, 1) #x.shape= (32, 32, 32, 48)
        self.g21 = gen_conv_block(2*cnum, 3, 2) #x.shape= (32, 16, 16, 48)
        self.g22 = gen_conv_block(4*cnum, 3, 1) #x.shape= (32, 16, 16, 96)
        self.g23 = gen_conv_block(4*cnum, 3, 1) #x.shape= (32, 16, 16, 96)
        self.g24 = gen_conv_block(4*cnum, 3, dilation_rate=2) #x.shape= (32, 16, 16, 96)
        self.g25 = gen_conv_block(4*cnum, 3, dilation_rate=4) #x.shape= (32, 16, 16, 96)
        self.g26 = gen_conv_block(4*cnum, 3, dilation_rate=8) #x.shape= (32, 16, 16, 96)
        self.g27 = gen_conv_block(4*cnum, 3, dilation_rate=16) #x.shape= (32, 16, 16, 96)
        # x_hallu1 = x 결과임

        # attention branch / conv_blocks1
        self.g28 = gen_conv_block(cnum, 5, 1) # g28 x.shape= (32, 64, 64, 24)
        self.g29 = gen_conv_block(cnum, 3, 2) # g29 x.shape= (32, 32, 32, 24)
        self.g30 = gen_conv_block(2*cnum, 3, 1) # g30 x.shape= (32, 32, 32, 48)
        self.g31 = gen_conv_block(4*cnum, 3, 2) # g31 x.shape= (32, 16, 16, 96)
        self.g32 = gen_conv_block(4*cnum, 3, 1) # g32 x.shape= (32, 16, 16, 96)
        self.g33 = gen_conv_block(4*cnum, 3, 1, activation=tf.keras.activations.relu) # (32, 16, 16, 96)

        # before pm1
        self.g34 = gen_conv_block(4*cnum, 3, 1) # g34 x.shape= (32, 16, 16, 96)
        self.g35 = gen_conv_block(4*cnum, 3, 1) # g35 x.shape= (32, 16, 16, 96)

        #추가 before conv3 blocks
        self.m3 = gen_conv_block(8*cnum, 3, 1) # (32, 16, 16, 192)

        # conv3 blocks # concat x.shape= (32, 16, 16, 192)  # 192 = 96 * 2
        self.g36 = gen_conv_block(4*cnum, 3, 1) # g36 x.shape= (32, 16, 16, 96)
        self.g37 = gen_conv_block(4*cnum, 3, 1) # g37 x.shape= (32, 16, 16, 96)
        self.g38 = gen_deconv_block(2*cnum)  # g38 x.shape= (32, 32, 32, 48)
        self.g39 = gen_conv_block(2*cnum, 3, 1) # g39 x.shape= (32, 32, 32, 48)
        self.g40 = gen_deconv_block(cnum)  # g40 x.shape= (32, 64, 64, 24)
        self.g41 = gen_conv_block(cnum//2, 3, 1) # g41 x.shape= (32, 64, 64, 12)
        self.g42 = gen_conv_block(3, 3, 1, activation=tf.keras.activations.tanh) # g42 x.shape= (32, 64, 64, 3)


        #추가 attention branch
        # conv_branch1
        self.g118 = gen_conv_block(cnum, 5, 1) #x.shape= (32, 64, 64, 24)
        self.g119 = gen_conv_block(cnum, 3, 2) #x.shape= (32, 32, 32, 24)
        self.g120 = gen_conv_block(2*cnum, 3, 1) #x.shape= (32, 32, 32, 48)
        self.g121 = gen_conv_block(2*cnum, 3, 1) #x.shape= (32, 32, 32, 48)
        self.g122 = gen_conv_block(4*cnum, 3, 1) #x.shape= (32, 32, 32, 96)
        self.g123 = gen_conv_block(4*cnum, 3, 1) #x.shape= (32, 32, 32, 96)
        self.g124 = gen_conv_block(4*cnum, 3, dilation_rate=2) #x.shape= (32, 32, 32, 96)
        self.g125 = gen_conv_block(4*cnum, 3, dilation_rate=4) #x.shape= (32, 32, 32, 96)
        self.g126 = gen_conv_block(4*cnum, 3, dilation_rate=8) #x.shape= (32, 32, 32, 96)
        self.g127 = gen_conv_block(4*cnum, 3, dilation_rate=16) #x.shape= (32, 32, 32, 96)
        # x_hallu2 = x 결과임

        # attention branch / conv_blocks2
        self.g128 = gen_conv_block(cnum, 5, 1) # g28 x.shape= (32, 64, 64, 24)
        self.g129 = gen_conv_block(cnum, 3, 2) # g29 x.shape= (32, 32, 32, 24)
        self.g130 = gen_conv_block(2*cnum, 3, 1) # g30 x.shape= (32, 32, 32, 48)
        self.g131 = gen_conv_block(4*cnum, 3, 1) # g31 x.shape= (32, 32, 32, 96) # stride를 2에서 1로 수정
        self.g132 = gen_conv_block(4*cnum, 3, 1) # g32 x.shape= (32, 32, 32, 96)
        self.g133 = gen_conv_block(4*cnum, 3, 1, activation=tf.keras.activations.relu) # (32, 32, 32, 96)

        #추가 before pm2
        self.g134 = gen_conv_block(4 * cnum, 3, 1)  # g34 x.shape= (32, 32, 32, 96)
        self.g135 = gen_conv_block(4 * cnum, 3, 1)  # g35 x.shape= (32, 32, 32, 96)

        # 추가 before conv4 blocks
        self.m4 = gen_conv_block(8 * cnum, 3, 1)  # (32, 32, 32, 192)

        # conv3 blocks # concat x.shape= (32, 32, 32, 192)  # 192 = 96 * 2
        self.g136 = gen_conv_block(4*cnum, 3, 1) # g36 x.shape= (32, 32, 32, 96)
        self.g137 = gen_conv_block(4*cnum, 3, 1) # g37 x.shape= (32, 32, 32, 96)
        #self.g138 = gen_deconv_block(2*cnum)  # g38 x.shape= (32, 32, 32, 48)
        self.g138 = gen_conv_block(2 * cnum, 3, 1)  # g39 x.shape= (32, 32, 32, 48)
        self.g139 = gen_conv_block(2*cnum, 3, 1) # g39 x.shape= (32, 32, 32, 48)
        self.g140 = gen_deconv_block(cnum)  # g40 x.shape= (32, 64, 64, 24)
        self.g141 = gen_conv_block(cnum//2, 3, 1) # g41 x.shape= (32, 64, 64, 12)
        self.g142 = gen_conv_block(3, 3, 1, activation=tf.keras.activations.tanh) # g42 x.shape= (32, 64, 64, 3)



    def call(self, x, mask):
        xin = x
        offset_flow1 = None
        offset_flow2 = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]

        x_noise = tf.keras.layers.GaussianNoise(stddev=0.1)(x)
        #To ones_x https://github.com/JiahuiYu/generative_inpainting/issues/40

        x_w_mask = tf.concat([x_noise, ones_x, ones_x*mask], axis=3)
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        #STAGE 1
        #print(xh, xw)
        #x_w_mask = tf.keras.layers.GaussianNoise(0.3)(x_w_mask)

        #BRANCH 0
        x = self.g0_1(x_w_mask)
        x = self.g0_2(x)
        x = self.g0_3(x)
        x = self.g0_4(x)
        x = self.g0_5(x)
        x = self.g0_6(x)
        mask_s0 = resize_mask_like(mask, x) # x.shape= (32, 32, 32, 96)
        x = self.g0_7(x)
        x = self.g0_8(x)
        x = self.g0_9(x)
        x = self.g0_10(x)
        x = self.g0_11(x)
        x = self.g0_12(x)
        x = self.g0_13(x)
        x_b0 = tf.image.resize(x, [xh, xw], method='bilinear')

          #BRANCH 1 
        x = self.g1_1(x_w_mask)
        x = self.g1_2(x)
        x = self.g1_3(x)
        x = self.g1_4(x)
        x = self.g1_5(x)
        x = self.g1_6(x)
        mask_s1 = resize_mask_like(mask, x) # x.shape= (32, 16, 16, 96)
        x = self.g1_7(x)
        x = self.g1_8(x)
        x = self.g1_9(x)
        x = self.g1_10(x)
        x = self.g1_11(x)
        x = self.g1_12(x)
        x = self.g1_13(x)
        x = tf.image.resize(x, [xh//2, xw//2], method='bilinear')
        x = self.g1_14(x)
        x = self.g1_15(x)
        x_b1 = tf.image.resize(x, [xh, xw], method='bilinear')
        #print(x_b1.shape)


          #BRANCH 2
        x = self.g2_1(x_w_mask)
        x = self.g2_2(x)
        x = self.g2_3(x)
        x = self.g2_4(x)
        x = self.g2_5(x)
        x = self.g2_6(x)
        x = self.g2_7(x)
        x = self.g2_8(x)
        x = self.g2_9(x)
        x = self.g2_10(x)
        x = self.g2_11(x)
        x = self.g2_12(x)
        x = self.g2_13(x)
        x = tf.image.resize(x, [xh//2, xw//2], method='bilinear')
        x = self.g2_14(x)
        x = self.g2_15(x)
        x_b2 = tf.image.resize(x, [xh, xw], method='bilinear')
        #print(x_b2.shape)

          #BRANCH 3
        x = self.g3_1(x_w_mask)
        x = self.g3_2(x)
        x = self.g3_3(x)
        x = self.g3_4(x)
        x = self.g3_5(x)
        x = self.g3_6(x)
        x = self.g3_7(x)
        x = self.g3_8(x)
        x = self.g3_9(x)
        x = self.g3_10(x)
        x = self.g3_11(x)
        x = self.g3_12(x)
        x = self.g3_13(x)
        x = tf.image.resize(x, [xh//2, xw//2], method='nearest')
        x = self.g3_14(x)
        x = self.g3_15(x)
        x = tf.image.resize(x, [xh, xw], method='nearest')
        x = self.g3_16(x)
        x_b3 = self.g3_17(x)
        #print(x_b3.shape)
        #neo
        x_merge = tf.concat([x_b0, x_b1, x_b2, x_b3], axis=3)

        x = self.m1(x_merge)
        x = self.m2(x)
        x = tf.clip_by_value(x, -1., 1.)
        x_stage1 = x
        #print("x_stage1.shape=", x_stage1.shape)



        #STAGE 2
        x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        xnow1 = x


        # conv branch
        x = self.g18(xnow1)
        x = self.g19(x)
        x = self.g20(x)
        x = self.g21(x)
        x = self.g22(x)    
        x = self.g23(x)    
        x = self.g24(x)
        x = self.g25(x)
        x = self.g26(x)
        x = self.g27(x)
        x_hallu1 = x


        # attention branch
        x = self.g28(xnow1)
        x = self.g29(x)
        x = self.g30(x)
        x = self.g31(x)
        x = self.g32(x)
        x = self.g33(x)
        #print("g33.shape=", x.get_shape())
        #print("shape of mask is:", mask_s.get_shape())


        x, offset_flow1 = contextual_attention(x, x, mask_s1, 3, 1, rate=2)
        x = self.g34(x)
        x = self.g35(x)
        #print("g35.shape=", x.get_shape())
        pm1 = x

        x = tf.concat([x_hallu1, pm1], axis=3)
        x = self.m3(x)
        x = tf.clip_by_value(x, -1., 1.)

        #print("before g36.shape=", x.get_shape())
        x = self.g36(x)
        x = self.g37(x)
        x = self.g38(x)
        x = self.g39(x)
        x = self.g40(x)
        x = self.g41(x)
        x = self.g42(x)

        x_stage2 = x
        #print("x_stage2.shape", x_stage2.shape)
        #print("offset_flow1.shape=", offset_flow1.shape)



        # STAGE 3
        x = x * mask + xin[:, :, :, 0:3] * (1. - mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        xnow2 = x


        # conv branch
        x = self.g118(xnow2)
        x = self.g119(x)
        x = self.g120(x)
        x = self.g121(x)
        x = self.g122(x)
        x = self.g123(x)
        x = self.g124(x)
        x = self.g125(x)
        x = self.g126(x)
        x = self.g127(x)
        x_hallu2 = x

        #추가
        x = self.g128(xnow2)
        x = self.g129(x)
        x = self.g130(x)
        x = self.g131(x)
        x = self.g132(x)
        x = self.g133(x)
        #print("g133.shape=", x.get_shape())
        #print("shape of mask_s0=", mask_s0.get_shape())

        x, offset_flow2 = contextual_attention(x, x, mask_s0, 3, 1, rate=2)
        x = self.g134(x)
        x = self.g135(x)
        #print("g135.shape=", x.get_shape())
        pm2 = x

        x = tf.concat([x_hallu2, pm2], axis=3)
        x = self.m4(x)
        x = tf.clip_by_value(x, -1., 1.)

        #print("before g136.shape=", x.get_shape())
        x = self.g136(x)
        x = self.g137(x)
        x = self.g138(x)
        x = self.g139(x)
        x = self.g40(x)
        x = self.g141(x)
        x = self.g142(x)

        x_stage3 = x
        #print("x_stage3.shape", x_stage3.shape)
        #print("offset_flow2.shape=", offset_flow2.shape)


        return x_stage1, x_stage2, x_stage3, offset_flow1, offset_flow2
    def model(self):
        x = tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
        mask = create_mask(FLAGS)
        return tf.keras.Model(inputs=[x], outputs=self.call(x,mask))

#GENERATOR NORMAL
class Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        cnum = 48

        self.g1 = gen_conv_block(cnum, 5, 1)
        self.g2 = gen_conv_block(2*cnum, 3, 2)
        self.g3 = gen_conv_block(2*cnum, 3, 1)
        self.g4 = gen_conv_block(4*cnum, 3, 2)
        self.g5 = gen_conv_block(4*cnum, 3, 1)
        self.g6 = gen_conv_block(4*cnum, 3, 1)
        self.g7 = gen_conv_block(4*cnum, 3, dilation_rate=2)
        self.g8 = gen_conv_block(4*cnum, 3, dilation_rate=4)
        self.g9 = gen_conv_block(4*cnum, 3, dilation_rate=8)
        self.g10 = gen_conv_block(4*cnum, 3, dilation_rate=16)
        self.g11 = gen_conv_block(4*cnum, 3, 1)
        self.g12 = gen_conv_block(4*cnum, 3, 1)
        self.g13 = gen_deconv_block(2*cnum)
        self.g14 = gen_conv_block(2*cnum, 3, 1)
        self.g15 = gen_deconv_block(cnum)
        self.g16 = gen_conv_block(cnum//2, 3, 1)
        self.g17 = gen_conv_block(3, 3, 1, activation=tf.keras.activations.tanh)
        self.g18 = gen_conv_block(cnum, 5, 1)
        self.g19 = gen_conv_block(cnum, 3, 2)
        self.g20 = gen_conv_block(2*cnum, 3, 1)
        self.g21 = gen_conv_block(2*cnum, 3, 2)
        self.g22 = gen_conv_block(4*cnum, 3, 1)
        self.g23 = gen_conv_block(4*cnum, 3, 1)
        self.g24 = gen_conv_block(4*cnum, 3, dilation_rate=2)
        self.g25 = gen_conv_block(4*cnum, 3, dilation_rate=4)
        self.g26 = gen_conv_block(4*cnum, 3, dilation_rate=8)
        self.g27 = gen_conv_block(4*cnum, 3, dilation_rate=16)
        self.g28 = gen_conv_block(cnum, 5, 1)
        self.g29 = gen_conv_block(cnum, 3, 2)
        self.g30 = gen_conv_block(2*cnum, 3, 1)
        self.g31 = gen_conv_block(4*cnum, 3, 2)
        self.g32 = gen_conv_block(4*cnum, 3, 1)
        self.g33 = gen_conv_block(4*cnum, 3, 1, activation=tf.keras.activations.relu)
        self.g34 = gen_conv_block(4*cnum, 3, 1)
        self.g35 = gen_conv_block(4*cnum, 3, 1)
        self.g36 = gen_conv_block(4*cnum, 3, 1)
        self.g37 = gen_conv_block(4*cnum, 3, 1)
        self.g38 = gen_deconv_block(2*cnum)
        self.g39 = gen_conv_block(2*cnum, 3, 1)
        self.g40 = gen_deconv_block(cnum)
        self.g41 = gen_conv_block(cnum//2, 3, 1)
        self.g42 = gen_conv_block(3, 3, 1, activation=tf.keras.activations.tanh)

    def call(self, x, mask, multi=0):
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)
         #To ones_x https://github.com/JiahuiYu/generative_inpainting/issues/40
        
        #STAGE 1
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.g4(x)
        x = self.g5(x)
        x = self.g6(x)
        mask_s = resize_mask_like(mask, x)
        x = self.g7(x)
        x = self.g8(x)
        x = self.g9(x)
        x = self.g10(x)
        x = self.g11(x)
        x = self.g12(x)
        x = self.g13(x)
        x = self.g14(x)
        x = self.g15(x)
        x = self.g16(x)
        x = self.g17(x)
        x_stage1 = x

        #STAGE 2
        x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        xnow = x
        # conv branch
        x = self.g18(xnow)    
        x = self.g19(x)
        x = self.g20(x)
        x = self.g21(x)
        x = self.g22(x)    
        x = self.g23(x)    
        x = self.g24(x)
        x = self.g25(x)
        x = self.g26(x)
        x = self.g27(x)
        x_hallu = x
        # attention branch
        x = self.g28(xnow)
        x = self.g29(x)
        x = self.g30(x)
        x = self.g31(x)
        x = self.g32(x)
        x = self.g33(x)
        #print("shape of x is:", x.get_shape())
        #print("shape of mask is:", mask_s.get_shape())
        x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
        x = self.g34(x)
        x = self.g35(x)
        pm = x
        x = tf.concat([x_hallu, pm], axis=3)

        x = self.g36(x)
        x = self.g37(x)
        x = self.g38(x)
        x = self.g39(x)
        x = self.g40(x)
        x = self.g41(x)
        x = self.g42(x)

        x_stage2 = x
        return x_stage1, x_stage2, offset_flow

#DISCRIMINATOR
class Discriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        cnum = 64

        self.dis_sn_conv_1 = dis_block(cnum)
        self.dis_sn_conv_2 = dis_block(cnum*2)
        self.dis_sn_conv_3 = dis_block(cnum*4)
        self.dis_sn_conv_4 = dis_block(cnum*4)
        self.dis_sn_conv_5 = dis_block(cnum*4)
        self.dis_sn_conv_6 = dis_block(cnum*4)
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, x):
        x = self.dis_sn_conv_1(x)
        x = self.dis_sn_conv_2(x)
        x = self.dis_sn_conv_3(x)
        x = self.dis_sn_conv_4(x)
        x = self.dis_sn_conv_5(x)
        x = self.dis_sn_conv_6(x)
        return self.flatten(x)
