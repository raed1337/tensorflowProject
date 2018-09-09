import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data" , one_hot=True)

num_node_l1=500
num_node_l2=500
num_node_l3=500


num_classes = 10

batch_size = 100

x= tf.placeholder('float')
y= tf.placeholder('float')

def neurol_network_model(data):
    hidden_layer_1 =  {'weights':tf.Variable(tf.random_normal([784 , num_node_l1])),
                       'biases':tf.Variable(tf.random_normal([num_node_l1]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_node_l1, num_node_l2])),
                      'biases': tf.Variable(tf.random_normal([num_node_l2]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([num_node_l2, num_node_l3])),
                      'biases': tf.Variable(tf.random_normal([num_node_l3]))}
    out_put_layer =     {'weights': tf.Variable(tf.random_normal([num_node_l3, num_classes])),
                      'biases': tf.Variable(tf.random_normal([num_classes]))}
    #( input_data * weights ) + biases===> to deal with problem if all feature = 0

    layer_1 = tf.add(tf.matmul(data , hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)
    out_put = tf.add(tf.matmul(layer_3, out_put_layer['weights']), out_put_layer['biases'])

    return  out_put


def train_model(x):
    predection = neurol_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predection,labels= y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)) :
                x_epochs , y_epoches = mnist.train.next_batch(batch_size)
                #c is the cost
                _ , c = sess.run([optimizer , cost] ,feed_dict={x: x_epochs , y:y_epoches})
                epoch_loss += c

                print('Epoch' , epoch , 'completed epochs of ' , num_epochs ,'with loss' , epoch_loss)

        correct = tf.equal(tf.argmax(predection,1) , tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy : ' , accuracy.eval({x:mnist.test.images , y:mnist.test.labels}))

train_model(x)