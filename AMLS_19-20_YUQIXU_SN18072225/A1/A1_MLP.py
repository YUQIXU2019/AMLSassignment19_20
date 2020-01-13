import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def allocate_weights_and_biases():

    # define number of hidden layers ..
    n_hidden_1 = 2048  # 1st layer number of neurons
    n_hidden_2 = 2048  # 2nd layer number of neurons
    
    
    
    # inputs placeholders
    X = tf.placeholder("float", [None, 68, 2])
    Y = tf.placeholder("float", [None, 2])  # 2 output classes
    #X = tf.keras.backend.placeholder("float", [None, 68, 2])
    #Y = tf.keras.backend.placeholder("float", [None, 2])  # 2 output classes
    
    # flatten image features into one vector (i.e. reshape image feature matrix into a vector)
    images_flat = tf.contrib.layers.flatten(X)  
    #images_flat = tf.keras.layers.Flatten(X)
    
    # weights and biases are initialized from a normal distribution with a specified standard devation stddev
    stddev = 0.01
    
    # define placeholders for weights and biases in the graph
    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, n_hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, 2], stddev=stddev))
    }

    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
    }
    
    return weights, biases, X, Y, images_flat
    

def multilayer_perceptron():
        
    weights, biases, X, Y, images_flat = allocate_weights_and_biases()

    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.math.sigmoid(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.math.sigmoid(layer_2)
    
    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer, X, Y



def MLP_A1(training_images, training_labels, test_images, test_labels, val_images, val_labels):
    # learning parameters
    learning_rate = 0.00001
    training_epochs = 500

    # display training accuracy every ..
    display_accuracy_step = 2


    #training_images, training_labels, test_images, test_labels = get_data()
    #training_images, training_labels, test_images, test_labels, val_images, val_labels =get_data_A1()
    logits, X, Y = multilayer_perceptron()

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # define training graph operation
    train_op = optimizer.minimize(loss_op)

    # graph operation to initialize all variables
    init = tf.global_variables_initializer()

    acc = np.zeros([500])
    val_acc = np.zeros([500])

    with tf.Session() as sess:
        # run graph weights/biases initialization op
        sess.run(init)
        # begin training loop ..
        for epoch in range(training_epochs):
            # run optimization operation (backprop) and cost operation (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images,
                                                               Y: training_labels})

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))

            ###############
            #if validation == True:
            #    _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images,Y: training_labels})

            if epoch % display_accuracy_step == 0:
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

                # calculate training accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                acc[epoch] = accuracy.eval({X: training_images, Y: training_labels})
                val_acc[epoch] = accuracy.eval({X: val_images, Y: val_labels})
                print("Accuracy: {:.3f}".format(accuracy.eval({X: training_images, Y: training_labels})))
                print("val_Accuracy: {:.3f}".format(accuracy.eval({X: val_images, Y: val_labels})))

        print("Optimization Finished!")

        # -- Define and run test operation -- #

        # apply softmax to output logits
        pred = tf.nn.softmax(logits)

        #  derive inffered calasses as the class with the top value in the output density function
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # run test accuracy operation ..
        print("Test Accuracy:", accuracy.eval({X: test_images, Y: test_labels}))
        test_acc = accuracy.eval({X: test_images, Y: test_labels})
        #plot acc and val_acc
        x = acc.nonzero()
        plt.plot(acc[x], label='accuracy',ls = '-')
        plt.plot(val_acc[x], label = 'val_accuracy',ls = '-')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('The graph of accuracy and val_accuracy (task A1)')
        plt.ylim([0.4, 1])
        plt.legend(loc='lower right')
        plt.show()

        return pred, acc[x], val_acc[x], test_acc




