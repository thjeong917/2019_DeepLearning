# 필요한 패키지들 
import os 
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array 
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import time

tf.set_random_seed(777)

def image_crop(img):
    
    cropped = []
        
    (img_h, img_w) = img.size
#     print(img.size)
 
    # crop 할 사이즈 : grid_w, grid_h
    grid_w = 32 # crop width
    grid_h = 32 # crop height
    range_w = (int)(img_w/grid_w)
    range_h = (int)(img_h/grid_h)
    # print(range_w, range_h)
 
    i = 0
 
    for w in range(range_w):
        for h in range(range_h):
            bbox = (h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            # print(h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            # 가로 세로 시작, 가로 세로 끝
            crop_img = img.crop(bbox)
            cropped.append(np.array(crop_img))
            
    return cropped

def crop2(img):
    (img_h, img_w) = img.size
    rand_h = random.randint(0,img_h-32)
    rand_w = random.randint(0,img_w-32)
#     print(rand_h, rand_w)
    
    bbox = (rand_h, rand_w, rand_h+32, rand_w+32)
    cropped = img.crop(bbox)
#     print(cropped)
#     plt.imshow(cropped)
    
    return np.array(cropped)

batch_size = 128

def random_crop(sess):
    data_list = glob('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab4_rnn/SR_dataset/291/*.bmp')
    data_list += glob('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab4_rnn/SR_dataset/291/*.jpg')

    # 이미지 모음
    # 순서를 먼저 np형 -> tf grayscale -> tf resize -> tf crop
    # train_list : 고화질 흑백(tensor image)
    # train_list_tf : 저화질 흑백(tensor image)
    cropped_img = []

    # 0~255 train set
    # 256~290 test set

    for path in data_list:
        cur = Image.open(path)
    #     cur = image_crop(cur)
    #     cropped_img += cur
        cur = crop2(cur)
        cropped_img.append(cur)


    cropped_img = np.array(cropped_img)
    np.random.shuffle(cropped_img)

    cropped_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    gray = tf.image.rgb_to_grayscale(cropped_placeholder)
    low = tf.image.resize_images(gray, (16, 16))
    low = tf.image.resize_images(low, (32, 32))

#     sess = tf.Session()
    x, y = sess.run([low, gray], feed_dict={cropped_placeholder:cropped_img})

    x_train = x[:256]
    x_test = x[256:]

    y_train = y[:256]
    y_test = y[256:]
    
    return x_train, x_test, y_train, y_test

sess = tf.Session()
x_train, x_test, y_train, y_test = random_crop(sess)

Wxh = tf.Variable(tf.random_normal([3, 3, 2, 32], stddev=0.01))
Wh = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
Wz = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
Whh = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
Why = tf.Variable(tf.random_normal([3, 3, 32, 1], stddev=0.01))

X = tf.placeholder(tf.float32, [None, None, None, 1])
Y = tf.placeholder(tf.float32, [None, None, None, 1])
h_ = tf.placeholder(tf.float32, [None, None, None, 32])

x0 = tf.concat([X, X], 3)
z0 = tf.nn.conv2d(x0, Wxh, strides=[1, 1, 1, 1], padding='SAME')
print(z0.shape)

h0_h = tf.nn.conv2d(h_, Wh, strides=[1, 1, 1, 1], padding='SAME')
h0_z = tf.nn.conv2d(z0, Wz, strides=[1, 1, 1, 1], padding='SAME')
print(h0_h.shape)
print(h0_z.shape)

h0 = tf.nn.relu(h0_h + h0_z)
r0 = tf.nn.relu(tf.nn.conv2d(h0, Whh, strides=[1, 1, 1, 1], padding='SAME'))
y0 = tf.nn.conv2d(r0, Why, strides=[1, 1, 1, 1], padding='SAME')

x1 = tf.concat([X, y0], 3)
z1 = tf.nn.conv2d(x1, Wxh, strides=[1, 1, 1, 1], padding='SAME')
h1_h = tf.nn.conv2d(h0, Wh, strides=[1, 1, 1, 1], padding='SAME')
h1_z = tf.nn.conv2d(z1, Wz, strides=[1, 1, 1, 1], padding='SAME')
h1 = tf.nn.relu(h1_h + h1_z)
r1 = tf.nn.relu(tf.nn.conv2d(h1, Whh, strides=[1, 1, 1, 1], padding='SAME'))
y1 = tf.nn.conv2d(r1, Why, strides=[1, 1, 1, 1], padding='SAME')


x2 = tf.concat([X, y1], 3)
z2 = tf.nn.conv2d(x2, Wxh, strides=[1, 1, 1, 1], padding='SAME')
h2_h = tf.nn.conv2d(h1, Wh, strides=[1, 1, 1, 1], padding='SAME')
h2_z = tf.nn.conv2d(z2, Wz, strides=[1, 1, 1, 1], padding='SAME')
h2 = tf.nn.relu(h2_h + h2_z)
r2 = tf.nn.relu(tf.nn.conv2d(h2, Whh, strides=[1, 1, 1, 1], padding='SAME'))
y2 = tf.nn.conv2d(r2, Why, strides=[1, 1, 1, 1], padding='SAME')

cost = tf.reduce_mean(tf.square(Y-y2) + tf.square(Y-y1) + tf.square(Y-y0))
tf.summary.scalar("Cost", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

accuracy = tf.reduce_mean(tf.image.psnr(y2, Y, max_val=255))
tf.summary.scalar("PSNR", accuracy)

SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

# init 필요함 모델 돌리기 전에
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/srcnn")
writer.add_graph(sess.graph)  # Show the graph

sess.run(tf.global_variables_initializer())

x_train, x_test, y_train, y_test = random_crop(sess)

hh = np.zeros((128, 32, 32, 32))
hh_test = np.zeros((35, 32, 32, 32))

# train_list = []
# train_list_low = [] 
start_vect=time.time()

for epoch in range(0, 10):
    avg_cost = 0
    x_train, x_test, y_train, y_test = random_crop(sess)
    
    total = len(x_train)
    
    for i in range(0, total, batch_size):
        batch_xs = x_train[i:i+batch_size]
        batch_ys = y_train[i:i+batch_size]
        feed_dict = {X: batch_xs, Y: batch_ys, h_: hh}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c
    
    psnr, summary = sess.run([accuracy, merged_summary], feed_dict={X: x_test, Y: y_test, h_: hh_test})
    writer.add_summary(summary, global_step=epoch)
    
    saver.save(sess, checkpoint_path, global_step=epoch)

    if epoch%500 == 0:
    	print('epoch:', epoch, '   cost:', c, '   PSNR:', psnr)
#     print('PSNR:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
    
print("training Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

set5_list = glob('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab4_rnn/SR_dataset/Set5/*.png')

# set5_gray = []
# set5_low = []

accuracy = tf.reduce_mean(tf.image.psnr(y2, Y, max_val=255))

# 5개 사진
i = 0
for path in set5_list:
    set5_gray = []
    set5_low = []
    
    cur = Image.open(path)
    cur_tf = tf.image.rgb_to_grayscale(cur)
    cur_np = sess.run(cur_tf)
    set5_gray.append(cur_np)

    (img_w, img_h, color) = cur_tf.get_shape().as_list()
#     print(img_w, img_h)
    cur_tf = tf.image.resize_images(cur_tf, (int(img_w/2), int(img_h/2)))
#     print(cur_tf.shape)
    cur_tf = tf.image.resize_images(cur_tf, (img_w, img_h))
#     print(cur_tf.shape)
    
    cur_np_low = sess.run(cur_tf)
    set5_low.append(cur_np_low)
    
    print(cur_np_low.shape)
    (a, b, c) = cur_np_low.shape
    
    h_test = np.zeros((1, a, b, 32))
    
    psnr_set5, set5_high = sess.run([accuracy, y2], feed_dict={X: set5_low, Y: set5_gray, h_: h_test})
    
    print('PSNR:', psnr_set5)
    
    tmp = np.reshape(set5_high, [set5_high.shape[1], set5_high.shape[2]])
    tmpimg = Image.fromarray(tmp)
    tmpimg = tmpimg.convert('L')
    tmpimg.save('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab4_rnn/SR_dataset/Set5/' + str(i) + '.jpg')
    
    i = i+1

