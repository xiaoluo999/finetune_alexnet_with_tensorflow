import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import cv2
import glob

weight_dict = np.load(r"F:\program\train_cnn-rnn-attention\pretrain\bvlc_alexnet.npy",encoding='bytes').item()
print(list(weight_dict.keys()))
for (key,value) in weight_dict.items():
    print(key)
    print(value[0].shape)
    print(value[1].shape)
height =227
width = 227
num_classes = 2
batch_size = 2
images = tf.placeholder(tf.float32,shape=[None,height,width,3])
labels = tf.placeholder(tf.float32,shape=[None,num_classes])
class data_input:
    def __init__(self,input_file,num_classes):
        with open(input_file) as f:
            lines = f.readlines()
            self.num = len(lines)
            random.shuffle(lines)
            self.data = []
            self.labels = []
            for line in lines:
                path = line.strip().split(",")[0]
                label = int(line.strip().split(",")[1])

                self.data.append(path)
                one_hot_label = np.zeros( shape=(num_classes,))
                one_hot_label[label] = 1.
                self.labels.append(one_hot_label)

    def get_next_batch(self,start,end,batch_size,num_classes):
        input = np.zeros(shape=[batch_size,height,width,3])
        labels = np.zeros(shape=[batch_size,num_classes])
        count = 0
        for i in range(start*batch_size,end*batch_size):
            img = cv2.imread(r"F:\data\dog_cat\train\dog\dog.0.jpg")
            img = cv2.resize(img, (width, height))
            img = img/255.0
            img = img-0.5
            img = img*2
            input[count,:,:,:] = img
            labels[count,:] = self.labels[i]
            count+=1
        return input,labels

data = data_input("train.txt",num_classes)
with slim.arg_scope([slim.conv2d],
                    padding = "VALID",
                    weights_initializer = tf.truncated_normal_initializer(stddev=0.005),
                    biases_initializer=tf.constant_initializer(0.1)):
    # TF中有两种作用域类型
    # 命名域(namescope)，通过tf.name_scope创建；
    # 变量域(variablescope)，通过tf.variable_scope创建；
    # 这两种作用域，对于使用tf.Variable()方式创建的变量，具有相同的效果，都会在变量名称前面，加上域名称。
    # 对于通过tf.get_variable()方式创建的变量，只有variablescope名称会加到变量名称前面，而namescope不会作为前缀。

    with tf.variable_scope("conv1"):
        weight1 = tf.Variable(initial_value=tf.zeros(shape=[11,11,3,96]))
        bias1 = tf.Variable(initial_value= tf.zeros(shape=[96]))
        conv1 = tf.nn.conv2d(images,weight1,strides=[1,4,4,1],padding="VALID")#55*55
        conv1 = tf.nn.relu(conv1)
        pool1 = slim.max_pool2d(conv1,[3,3],stride=2)#27*27
    with tf.variable_scope("conv2"):
        weight2 = tf.Variable(initial_value=tf.zeros(shape=[5, 5, 48, 256]))#输入节点为2*48=96
        bias2 = tf.Variable(initial_value=tf.zeros(shape=[256]))
        input_groups = tf.split(axis=3, num_or_size_splits=2, value=pool1)#将一个张量切成两份
        weight_groups = tf.split(axis=3, num_or_size_splits=2,value=weight2)
        output_groups = [tf.nn.conv2d(i, k,strides=[1, 1, 1, 1],padding="SAME")
                         for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv2 = tf.concat(axis=3, values=output_groups)

        conv2 = tf.nn.relu(conv2)
        pool2 = slim.max_pool2d(conv2,[3,3],stride=2)
    with tf.variable_scope("conv3"):
        weight3 = tf.Variable(initial_value=tf.zeros(shape=[3,3,256,384]))
        bias3 = tf.Variable(initial_value= tf.zeros(shape=[384]))
        conv3 = tf.nn.conv2d(pool2,weight3,strides=[1,1,1,1],padding="SAME")
        conv3 = tf.nn.relu(conv3)
    with tf.variable_scope("conv4"):
        weight4 = tf.Variable(initial_value=tf.zeros(shape=[3, 3, 192, 384]))#输入节点为384
        bias4 = tf.Variable(initial_value=tf.zeros(shape=[384]))
        input_groups = tf.split(axis=3, num_or_size_splits=2, value=conv3)  # 将一个张量切成两份
        weight_groups = tf.split(axis=3, num_or_size_splits=2, value=weight4)
        output_groups = [tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding="SAME")
                         for i, k in zip(input_groups, weight_groups)]
        # Concat the convolved output together again
        conv4 = tf.concat(axis=3, values=output_groups)
        conv4 = tf.nn.relu(conv4)
    with tf.variable_scope("conv5"):
        weight5 = tf.Variable(initial_value=tf.zeros(shape=[3, 3, 192, 256]))#输入节点为384
        bias5 = tf.Variable(initial_value=tf.zeros(shape=[256]))
        input_groups = tf.split(axis=3, num_or_size_splits=2, value=conv4)  # 将一个张量切成两份
        weight_groups = tf.split(axis=3, num_or_size_splits=2, value=weight5)
        output_groups = [tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding="SAME")
                         for i, k in zip(input_groups, weight_groups)]
        # Concat the convolved output together again
        conv5 = tf.concat(axis=3, values=output_groups)
        conv5 = tf.nn.relu(conv5)
        pool5 = slim.max_pool2d(conv5, [3, 3], stride=2,padding="VALID")
    with tf.variable_scope("fc1"):
        pool5 = tf.layers.flatten(pool5)
        weight6 = tf.Variable(initial_value=tf.zeros(shape=[pool5.shape[-1],4096]))
        bias6 = tf.Variable(initial_value= tf.zeros(shape=[4096]))
        fc_1 = tf.nn.relu(tf.matmul(pool5,weight6)+bias6)
    with tf.variable_scope("fc2"):
        weight7 = tf.Variable(initial_value=tf.zeros(shape=[4096, 4096]))
        bias7 = tf.Variable(initial_value=tf.zeros(shape=[4096]))
        fc_2 = tf.nn.relu(tf.matmul(fc_1, weight7) + bias7)
    with tf.variable_scope("fc3"):
        weight8 = tf.Variable(initial_value=tf.zeros(shape=[4096, num_classes]))
        bias8 = tf.Variable(initial_value=tf.zeros(shape=[num_classes]))
        out = tf.matmul(fc_2, weight8) + bias8

        print(out)
#optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = tf.train.AdamOptimizer(0.0001)
#loss = tf.reduce_mean(tf.reduce_sum(-labels*tf.log(out),-1))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=out))
#acc = tf.reduce_mean(tf.equal(tf.arg_max(labels,-1),tf.arg_max(out,tf.arg_max(out))))
train_var_list = []
for var in tf.trainable_variables():
    temp = str(var.name)
    if temp.startswith("fc"):
        train_var_list.append(var)


train_op = optimizer.minimize(loss,var_list=train_var_list)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(tf.assign(weight1,weight_dict["conv1"][0]))
sess.run(tf.assign(bias1,weight_dict["conv1"][1]))
sess.run(tf.assign(weight2,weight_dict["conv2"][0]))
sess.run(tf.assign(bias2,weight_dict["conv2"][1]))

sess.run(tf.assign(weight3,weight_dict["conv3"][0]))
sess.run(tf.assign(bias4,weight_dict["conv3"][1]))
sess.run(tf.assign(weight4,weight_dict["conv4"][0]))
sess.run(tf.assign(bias4,weight_dict["conv4"][1]))
sess.run(tf.assign(weight5,weight_dict["conv5"][0]))
sess.run(tf.assign(bias5,weight_dict["conv5"][1]))


for i in range(data.num//batch_size-1):
    batch_x ,batch_y = data.get_next_batch(i,i+1,batch_size,num_classes)
    # print(np.any(np.isnan(batch_x)))
    # print(np.any(np.isnan(batch_y)))
    _,res,ls= sess.run([train_op,out,loss],feed_dict={images:batch_x,labels:batch_y})
    if i%10 ==0:
        print("batch %d:loss %f"%(i,ls))
    #print(res)
    #break
