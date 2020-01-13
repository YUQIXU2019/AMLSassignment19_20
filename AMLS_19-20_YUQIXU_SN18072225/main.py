import csv
import pandas as pd
import splitdata
from A1 import A1_SVM, A1_MLP
from A2 import A2_SVM, A2_MLP
from B1 import B1_MLP, B1_NN
from B2 import B2_CNN, B2_NN
import pickle


if __name__ == '__main__':

    csv_data = pd.read_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/celeba/labels.csv', sep='\t', encoding='utf-8',
                           index_col=False)
    csv_data.drop('Unnamed: 0', axis=1, inplace=True)
    csv_data.to_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/celeba/new_labels.csv', index=False)

    csv_data = pd.read_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/cartoon_set/labels.csv', sep='\t', encoding='utf-8', index_col=False)
    csv_data.drop('Unnamed: 0', axis=1, inplace=True)
    csv_data = csv_data[['file_name', 'eye_color', 'face_shape']]
    csv_data.to_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/cartoon_set/new_labels.csv', index=False)

    csv_data = pd.read_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/celeba_test/labels.csv', sep='\t', encoding='utf-8',
                           index_col=False)
    csv_data.drop('Unnamed: 0', axis=1, inplace=True)
    csv_data.to_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/celeba_test/new_labels.csv', index=False)

    csv_data = pd.read_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/cartoon_set_test/labels.csv', sep='\t', encoding='utf-8', index_col=False)
    csv_data.drop('Unnamed: 0', axis=1, inplace=True)
    csv_data = csv_data[['file_name', 'eye_color', 'face_shape']]
    csv_data.to_csv('/Users/wyl/Desktop/AMLS_19-20_SN12345678/cartoon_set_test/new_labels.csv', index=False)

    # ======================================================================================================================
    # Data preprocessing
    A1_tr_X, A1_tr_Y, A1_te_X, A1_te_Y = splitdata.get_data()
    #training_images, training_labels, test_images, test_labels, val_images, val_labels = splitdata.get_data_A1()

    #A1_add_te_X, A1_add_te_Y = splitdata.get_test_data()
    #add_test_images, add_test_labels = splitdata.get_test_data_A1()
    #A2
    #A2_tr_X, A2_tr_Y, A2_te_X, A2_te_Y = splitdata.get_data_2()
    #training_images_A2, training_labels_A2, test_images_A2, test_labels_A2, val_images_A2, val_labels_A2 = splitdata.get_data_A2()

    #A2_add_te_X, A2_add_te_Y = splitdata.get_test_data_2()
    #add_test_images_A2, add_test_labels_A2 = splitdata.get_test_data_A2()

    #B1
    #training_images_B1, training_labels_B1, test_images_B1, test_labels_B1, val_images_B1, val_labels_B1 = splitdata.get_data_B1()

    #B2
    #training_images_B2, training_labels_B2, test_images_B2, test_labels_B2, val_images_B2, val_labels_B2 = splitdata.get_data_B2()
    #add_test_images_B2, add_test_labels_B2 = splitdata.get_test_data_B2()
    #print(len(add_test_images_B2))
    #print(len(add_test_labels_B2))
    # ======================================================================================================================

    # Task A1
    # SVM for task A1
    pred_A1_SVM, acc_A1_test_SVM, val_acc_A1_SVM = A1_SVM.img_SVM(A1_tr_X.reshape((3000, 68*2)), list(zip(*A1_tr_Y))[0], A1_te_X.reshape((len(A1_te_Y), 68*2)), list(zip(*A1_te_Y))[0])
    #acc_A1_train_SVM = val_acc_A1_SVM.mean()
    # MLP for task A1
    #pred_A1_MLP, acc_A1_train_MLP, val_acc_A1_MLP, acc_A1_test_MLP = A1_MLP.MLP_A1(training_images, training_labels, add_test_images, add_test_labels, val_images, val_labels)
    #acc_A1_train_MLP = acc_A1_train_MLP[-1]
    # Clean up memory/GPU etc...             # Some code to free memory if necessary.

    # ======================================================================================================================

    # Task A2
    #pred_A2_SVM, acc_A2_test_SVM, val_acc_A2_SVM = A2_SVM.img_SVM(A2_tr_X.reshape((3000, 68*2)), list(zip(*A2_tr_Y))[0], A2_add_te_X.reshape((969, 68*2)), list(zip(*A2_add_te_Y))[0])
    #acc_A2_train_SVM = val_acc_A2_SVM.mean()
    # MLP for task A1
    #pred_A2_MLP, acc_A2_train_MLP, val_acc_A2_MLP, acc_A2_test_MLP = A2_MLP.MLP_A2(training_images_A2,
    #                                                                               training_labels_A2,
    #                                                                               add_test_images_A2, add_test_labels_A2,
    #                                                                               val_images_A2, val_labels_A2)

    #acc_A2_train_MLP = acc_A2_train_MLP[-1]
    # Clean up memory/GPU etc...             # Some code to free memory if necessary.

    # ======================================================================================================================

    # Task B1
    #pred_B1_MLP, acc_B1_train_MLP, val_acc_B1_MLP, acc_B1_test_MLP = B1_MLP.MLP_B1(training_images_B1,
    #                                                                               training_labels_B1, test_images_B1,
    #                                                                               test_labels_B1, val_images_B1,
    #                                                                               val_labels_B1)

    #pred_B1_NN, acc_B1_train_NN, val_acc_B1_NN, acc_B1_test_NN = B1_NN.B1_NN(training_images_B1,
    #                                                                               training_labels_B1, test_images_B1,
    #                                                                               test_labels_B1, val_images_B1,
    #                                                                               val_labels_B1)
    #acc_B1_train_MLP = acc_B1_train_MLP[-1]
    # Clean up memory/GPU etc...             # Some code to free memory if necessary.

    # ======================================================================================================================

    # Task B2
    #pred_B2_CNN, acc_B2_train_CNN, val_acc_B2_CNN, acc_B2_test_CNN = B2_CNN.B2_CNN(training_images_B2,
    #                                                                               training_labels_B2, add_test_images_B2,
    #                                                                               add_test_labels_B2, val_images_B2,
    #                                                                               val_labels_B2)

    #pred_B2_MLP, acc_B2_train_MLP, val_acc_B2_MLP, acc_B2_test_MLP = B2_NN.B2_NN(training_images_B2,
    #                                                                               training_labels_B2, test_images_B2,
    #                                                                               test_labels_B2, val_images_B2,
    #                                                                               val_labels_B2)
    # acc_B2_train_MLP = acc_B2_train_MLP[-1]

    ## Print out your results with following format:
    #print(acc_A1_test_SVM)
    #print(acc_A1_train_SVM)
    # print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
    # acc_A2_train, acc_A2_test,
    # acc_B1_train, acc_B1_test,
    # acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'