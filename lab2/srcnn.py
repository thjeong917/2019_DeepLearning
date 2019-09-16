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
    data_list = glob('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab2_cnn/SR_dataset/291/*.bmp')
    data_list += glob('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab2_cnn/SR_dataset/291/*.jpg')

    # 이미지 모음
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

#     print(low)

#     sess = tf.Session()
    x, y = sess.run([low, gray], feed_dict={cropped_placeholder:cropped_img})

#     print(len(x), len(y))

    x_train = x[:256]
    x_test = x[256:]

    y_train = y[:256]
    y_test = y[256:]
    
    return x_train, x_test, y_train, y_test


sess = tf.Session()
x_train, x_test, y_train, y_test = random_crop(sess)
# len(y_train)

# X = tf.placeholder(tf.float32, [None, 1024])
# X_img = tf.reshape(X, [-1,32,32,1])

# X = tf.placeholder(tf.float32, [None, 32, 32, 1])
# Y = tf.placeholder(tf.float32, [None, 32, 32, 1])

X = tf.placeholder(tf.float32, [None, None, None, 1])
Y = tf.placeholder(tf.float32, [None, None, None, 1])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01))
b1 = tf.Variable(tf.random_normal([64]))
#    Conv     -> (?, 28, 28, 64)
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
L1 = tf.nn.relu(L1)

W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
b2 = tf.Variable(tf.random_normal([64]))

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
L2 = tf.nn.relu(L2)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 1]))
b3 = tf.Variable(tf.random_normal([1]))

L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
# L3 -> (?, 32, 32, 1)

cost = tf.reduce_mean(tf.square(Y-L3))
tf.summary.scalar("Cost", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

accuracy = tf.reduce_mean(tf.image.psnr(L3, Y, max_val=255))
tf.summary.scalar("PSNR", accuracy)


# init 필요함 모델 돌리기 전에
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/srcnn")
writer.add_graph(sess.graph)  # Show the graph

sess.run(tf.global_variables_initializer())

# train_list = []
# train_list_low = [] 
start_vect=time.time()

for epoch in range(0, 10000):
    avg_cost = 0
    x_train, x_test, y_train, y_test = random_crop(sess)
    
    total = len(x_train)
    
    for i in range(0, total, batch_size):
        batch_xs = x_train[i:i+batch_size]
        batch_ys = y_train[i:i+batch_size]
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c
    
    psnr, summary = sess.run([accuracy, merged_summary], feed_dict={X: x_test, Y: y_test})
    writer.add_summary(summary, global_step=epoch)
    
    print('epoch:', epoch, '   cost:', c, '   PSNR:', psnr)
#     print('PSNR:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
    
print("training Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))


set5_list = glob('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab2_cnn/SR_dataset/Set5/*.png')

accuracy = tf.reduce_mean(tf.image.psnr(L3, Y, max_val=255))

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
    
    psnr_set5, set5_high = sess.run([accuracy, L3], feed_dict={X: set5_low, Y: set5_gray})
    
    print('PSNR:', psnr_set5)
    
    tmp = np.reshape(set5_high, [set5_high.shape[1], set5_high.shape[2]])
    tmpimg = Image.fromarray(tmp)
    tmpimg = tmpimg.convert('L')
    tmpimg.save('/Users/taehwa/Desktop/TAEHWA/2019-1학기/딥러닝/실습/lab2_cnn/SR_dataset/Set5/' + str(i) + '.jpg')
    
    i = i+1