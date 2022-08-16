# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:56:34 2019

@author: 14342
"""
# first neural network with keras tutorial
import os
#os.chdir('/Users/esestaff/Documents/GitHub/Hierarchical-DNN-Activity-Recognition')

#My modification in order to work on my computer 
#os.chdir('~/Documentos/WorkspaceProjetos/datasets/Extrasensory/data')

import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from Extrasensory_Manipulation import *
from Inputs_HDLAct import *
import glob
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# - data that is passed should have all the labels (parent and child level) ready
def data_cleaner(dataset, feature_set_range, parent_labels, child_labels=None):
    """This function cleans the data given to it and produced X and y for the
    parent and child levels that can be used to train classifiers.
    
    Input:
        featur_set_range[list]: the column indices of features to be used for our models
        parent_labels[list]: list of the parent labels
        child_labels[list of list]: a list of list of the child labels. Each element
                                    is the list of one parent branch
        dataset[pd dataframe]: the dataframe to extract X and y from
    
    Output:
        X_parent[pd dataframe]: features of the parent level model
        y_parent[pd dataframe]: response of the parent level model
        X_child[list(pd dataframe)]: list of the features for the child level models
        y_child[list(pd dataframe)]: list of the responses for the child level models
    """
    
    num_parents = len(parent_labels)       # Number of the classes at parent level
    if child_labels is not None:
        assert len(parent_labels) == len(child_labels)
    
    # Cutting out the features and labels from the entire dataset
    features = dataset.iloc[:,feature_set_range]  #add fillna after this line
    labels = dataset[parent_labels]
    
    # Consider eliminating the for loops
    if child_labels is not None:
        for i in range(num_parents):
            for j in range(len(child_labels[i])):
                labels = pd.concat([labels,dataset[child_labels[i][j]]], axis=1)
    
    # Junta os labels com as features para formar o dataset que será utilizado nos modelos
    raw_data = pd.concat([features,labels],axis=1)
    #Retira do dataset as linhas que contenham elementos que não são números
    raw_data = raw_data.dropna()
    
    # We are making sure that we have samples that belong to exactly one class
    # of parent labels at the same time. As the parent labels are chosen so that
    # to be completely exclusive, we expect to get none of such samples
    
    # Parent level:
    raw_data['Parent_Belonging'] = 0   # Number of parent classes a sample belongs to
    
    for i in range(num_parents):
        raw_data['Parent_Belonging'] += (raw_data[parent_labels[i]] == 1) * 1
    
    #Child level:
    if child_labels is not None:
        raw_data['Child_Belonging'] = 0    # Number of child classes a sample belongs to

        for i in range(num_parents):
            for j in range(len(child_labels[i])):
                raw_data['Child_Belonging'] += (raw_data[child_labels[i][j]] == 1)*1
        raw_data = raw_data[raw_data['Child_Belonging'] == 1]
        
    raw_data = raw_data[raw_data['Parent_Belonging'] == 1]
    
    
    print('\n\nNumber of samples remaining after removing missing features and labels: ',len(raw_data))
    print()
    
    # Adding the parent label column to the data
    raw_data['Parent_label_index'] = 0    # The index of the parent class that each sample belongs to
    
    for i in range(num_parents):
        raw_data['Parent_label_index'] += (i+1)*(1*(raw_data[parent_labels[i]] == 1))
    
    raw_data['Parent_label_index'] -= 1

    
    if child_labels is not None:
        raw_data['Child_label_index'] = 0     # The index of the child class that each sample belongs to

        for i in range(num_parents):
            for j in range(len(child_labels[i])):
                raw_data['Child_label_index'] += (j+1)*(1*(raw_data[child_labels[i][j]] == 1))
        raw_data['Child_label_index'] -= 1
        
    
    # Collecting X and y for the training phase of the classifiers
    X_parent = raw_data.iloc[:,range(0, len(feature_set_range))]
    X_parent = preprocessing.scale(X_parent, axis=0)
    y_parent = raw_data['Parent_label_index']
    
    print('Descrição do dataset')
    print(raw_data.shape)
    print(raw_data.columns)
    
    if child_labels is not None:
        
        X_child = []
        y_child = []
        for i in range(num_parents):
            dataset_ith_parent = raw_data[raw_data['Parent_label_index'] == i]
            X_child.append(dataset_ith_parent.iloc[:,range(0, len(feature_set_range))])
            X_child[i] = preprocessing.scale(X_child[i], axis=0)
            y_child.append(dataset_ith_parent['Child_label_index'])
    
        return(X_parent, y_parent, X_child, y_child)
    else:
        return(X_parent, y_parent)
        
def return_accuracy(clf):
    #y_pred = clf.predict_classes(X_test) #Esta função foi descontinuada
    y_pred = clf.predict(X_test)
    classes_x = np.argmax(y_pred,axis=1)
    print(classes_x)
    f1_accuracy['flat'] = f1_score(y_test.values, classes_x, average='macro')
    BA_accuracy['flat'] = balanced_accuracy_score(y_test.values, classes_x)
    accuracy['flat'] = accuracy_score(y_test.values, classes_x)
    return accuracy, f1_accuracy, BA_accuracy
 
def return_accuracy_shallow(clf, model_name):
    y_pred = clf.predict(X_test)
    f1_accuracy[model_name] = f1_score(y_test.values, y_pred, average='macro')
    BA_accuracy[model_name] = balanced_accuracy_score(y_test.values, y_pred)
    accuracy[model_name] = accuracy_score(y_test.values, y_pred)
    return accuracy, f1_accuracy, BA_accuracy

    ##########################################################################
    #--------------------------| Main Program |------------------------------#
    ##########################################################################
    
def print_sep():
    print('***************************************************************************\n')

if __name__ == '__main__':
    
    #Dicionários com o resultado de todos os modelos implementados
    f1_accuracy = {}
    BA_accuracy = {}
    accuracy = {}
       

    # Retorna o conjunto de todos os usuários como um dicionário
    dataset_uuids = readdata_csv(data_dir) 
    # Retorna a UUID de todos os usuários
    uuids = list(dataset_uuids.keys())
    
    #Combining the all users' data to dataset
    dataset = dataset_uuids[uuids[0]]
    #Junção de todos os datasets
    for i in range(1,len(uuids)):
        dataset = pd.concat([dataset, dataset_uuids[uuids[i]]],axis=0)
    
    # Retorna uma lista que contém todos os sensores com seus ranges 
    sensors_list = sensors()
    feature_set_range = []
    
    for i in range(len(sensors_to_use)):
        feature_set_range += sensors_list[sensors_to_use[i]]
    
    print("\n\nFeature set \n\n")
    print(feature_set_range)
    print_sep()


#    ###########################################################################
#    #-----------------------|  w/o Hierarchy|-----------------------------#
#    ###########################################################################
    print('DNN Flat')
    print_sep()
    #Definição dos Labels que serão utilizados para avaliar a performance do modelo
    parent_labels = ['label:OR_standing','label:SITTING','label:LYING_DOWN','label:FIX_running','label:FIX_walking','label:BICYCLING']
    
    # Esta função separa os conjuntos X e y de acordo com os labels indicados em parent_labels
    X_parent, y_parent = data_cleaner(dataset, feature_set_range, parent_labels)
    # Separação dos conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_parent, y_parent, test_size=test_split)


    best_model_path = './model.hdf5'
    #Callback para salvar o modelo e os pesos em um arquivo especificado pelo filepath
    checkpointer = ModelCheckpoint(filepath=best_model_path, verbose=1, save_best_only=True)
    

#    ###########################################################################
#    #-----------------------| DNN|-----------------------------#
#    ###########################################################################

    # Criação do Modelo de rede neural
    clf = Sequential()
    clf.add(Dense(128, input_dim=len(feature_set_range), activation='relu'))
    clf.add(Dropout(drop_out_per))
    #clf.add(Dense(512, activation='relu'))
    #clf.add(Dropout(0.4))
    clf.add(Dense(64, activation='relu'))
    clf.add(Dense(6, activation='softmax'))
    
    #Segundo modelo de Rede Neural
    clf_temp = Sequential()
    clf_temp.add(Dense(128, input_dim=len(feature_set_range), activation='relu'))
    clf_temp.add(Dropout(drop_out_per))
    #clf.add(Dense(512, activation='relu'))
    #clf.add(Dropout(0.4))
    clf_temp.add(Dense(64, activation='relu'))
    clf_temp.add(Dense(6, activation='softmax'))
    
    clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #clf.fit(X_train, y_train, batch_size=nb_batch,  epochs=nb_epoch, validation_split = val_split, class_weight='balanced', callbacks=[checkpointer])
    clf.fit(X_train, y_train, batch_size=nb_batch,  epochs=nb_epoch, validation_split = val_split, class_weight=None, callbacks=[checkpointer])
    
    clf_temp.load_weights(best_model_path)
    clf_temp.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #return_accuracy(clf_temp)
    print_sep()
    print('Resultados Rede 1')
    print(return_accuracy(clf))
    print_sep()
    print('Resultados Rede 2')
    print(return_accuracy(clf_temp))
    
    y_pred = clf.predict(X_test)
    classes__ = np.argmax(y_pred, axis=1)
    confusion_flat = confusion_matrix(y_test.values, classes__)
    
    print('Matriz de Confusão Modelo Flat')
    print(confusion_flat)
    print_sep()

#    ###########################################################################
#    #-----------------------| Baselines|-----------------------------#
#    ###########################################################################
    #Modelos de base para realizar a comparação entre a DNN do tipo flat
    functions = [tree.DecisionTreeClassifier(max_depth = 20), KNeighborsClassifier(n_neighbors = 10), svm.SVC(kernel = 'rbf'), \
             RandomForestClassifier(n_estimators = 10, max_depth = 10), MLPClassifier(hidden_layer_sizes = (64), max_iter = 100)] #MultinomialNB(alpha = 1.0, class_prior = None, fit_prior = True)]

    #Treinamento e avaliação dos modelos de comparação
    for clf in functions:
        model_name = str(clf).split('(')[0]
        print(model_name)
        clf.fit(X_train, y_train)
        #Calcula a acurácia, o f1-score e a acurácia balanceada de cada modelo
        return_accuracy_shallow(clf, model_name)
    
    #Imprime a acurácia para cada modelo
    print('Resultados Modelos de Base')
    print(accuracy)
    print('***************************************************************************************\n\n')
    #Imprime o F1-Score dos
    print(f1_accuracy)
    print('***************************************************************************************\n\n')
    print(BA_accuracy)
    print('***************************************************************************************\n\n')
        #return f1_accuracy, BA_accuracy, accuracy
    #    f1_acc = pd.DataFrame.from_dict(list(f1_accuracy.items()))
    #    f1_acc.to_csv('./results/f1.csv')
    #    
    #    import pickle
    #    curr_ckpt = util.load_ckpt('./model')
    #    loaded_model = pickle.load(open('./model', 'rb'))
    #    y_pred = loaded_model.predict_classes(X_test)
    #    result = balanced_accuracy_score(y_test.values, y_pred)
    #    print(result)

#    
#    ###########################################################################
#    #-----------------------| DNN  with Hierarchy|----------------------------#
#    ###########################################################################

    #We don't have these labels so we add them by our own Based on the child labels defined
    
    parent_labels = ['Stationary','NonStationary']          #Labels das classes pai para o primeiro nível na hierarquia
    
    child_labels = [['label:OR_standing','label:SITTING','label:LYING_DOWN'],\
                    ['label:FIX_running','label:FIX_walking','label:BICYCLING']]        #Labels das classes filhas para o segundo nível na hierarquia
    
    
    # Adicionando as linhas das classes filhas às classes pai (Estacionária)
    dataset['Stationary'] = np.logical_or(dataset['label:OR_standing'],dataset['label:SITTING'])
    dataset['Stationary'] = np.logical_or(dataset['Stationary'],dataset['label:LYING_DOWN'])*1
    
    # Adicionando as linhas das classes filhas às classes pai (Estacionária)
    dataset['NonStationary'] = np.logical_or(dataset['label:FIX_running'],dataset['label:FIX_walking'])
    dataset['NonStationary'] = np.logical_or(dataset['NonStationary'],dataset['label:BICYCLING'])*1
       
    X_parent, y_parent, X_child, y_child = data_cleaner(dataset, feature_set_range, parent_labels, child_labels)

    X_train, X_test, y_train, y_test = train_test_split(X_parent, y_parent, test_size=test_split)

    #-------------------------------------------------------------------------#

    clf = Sequential()
    clf.add(Dense(128, input_dim=len(feature_set_range), activation='relu'))
    clf.add(Dropout(drop_out_per))
    #clf.add(Dense(512, activation='relu'))
    #clf.add(Dropout(0.4))
    clf.add(Dense(64, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
     
    best_model_path = './model_parent.hdf5'
    checkpointer = ModelCheckpoint(filepath = best_model_path, verbose=1, save_best_only=True)
    
    clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    clf.fit(X_train, y_train, batch_size=nb_batch, epochs=nb_epoch, validation_split = val_split, class_weight = None)#, callbacks=[checkpointer])
    
    y_pred = clf.predict(X_test)
    classes__ = np.argmax(y_pred, axis=1)

    f1_accuracy['parent'] = f1_score(y_test.values, classes__, average='macro')
    BA_accuracy['parent'] = balanced_accuracy_score(y_test.values, classes__)
    accuracy['parent'] = accuracy_score(y_test.values, classes__)
    
    confusion_flat = confusion_matrix(y_test.values, classes__)
    
    print_sep()
    print('RESULTADOS DA CLASSE PAI')
    
    print('Acurácia: ', accuracy['parent'])
    print()
    print('BA_Acurácia:',BA_accuracy['parent'])
    print()
    print('F1-Score:',f1_accuracy['parent'])
    
    print('Matriz de Confusão classes Pai')
    print(confusion_flat)
    
    #-------------------------------------------------------------------------#
    clf_child = []
    y_child_pred = []
    confusion_child = []
    
    for i in range(len(parent_labels)):
        X_train, X_test, y_train, y_test = \
        train_test_split(X_child[i], y_child[i], test_size=test_split)
        
        clf = Sequential()
        clf.add(Dense(128, input_dim=len(feature_set_range), activation='relu'))
        clf.add(Dropout(drop_out_per))
        #clf.add(Dense(512, activation='relu'))
        #clf.add(Dropout(0.4))
        clf.add(Dense(64, activation='relu'))
        clf.add(Dense(len(child_labels[i]), activation='softmax'))
        
        clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        best_model_path = './model_{}.hdf5'.format(i)
        checkpointer = ModelCheckpoint(filepath=best_model_path, verbose=1, save_best_only=True)
    
        
        clf.fit(X_train, y_train, batch_size=nb_batch, validation_split = val_split, epochs=nb_epoch, class_weight = None)#, callbacks=[checkpointer])
        
        y_pred = clf.predict(X_test)
        classes__ = np.argmax(y_pred,axis=1)
        
        confusion_flat = confusion_matrix(y_test.values, classes__)
 
        f1_accuracy['child_{}'.format(i)] = f1_score(y_test.values, classes__, average='macro')
        BA_accuracy['child_{}'.format(i)] = balanced_accuracy_score(y_test.values, classes__)
        accuracy['child_{}'.format(i)] = accuracy_score(y_test.values, classes__)
        
        print_sep()
        print('RESULTADOS DA CLASSES FILHAS')
    
        print('Acurácia: ', accuracy['child_{}'.format(i)])
        print()
        print('BA_Acurácia:', BA_accuracy['child_{}'.format(i)])
        print()
        print('F1-Score:', f1_accuracy['child_{}'.format(i)])
        print('Matriz de Confusão modelo {}'.format(i))
        print(confusion_flat)


#Impressão da matriz de confusão

     
    #conf_mtrx = confusion_matrix(y_parent_test.values, y_parent_pred.round())
    
#    df_cm = pd.DataFrame(confusion_matrix, range(2),range(2))
#    plt.figure(figsize = (10,7))
#    sn.set(font_scale=1.4)#for label size
#    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
#    f1_score(y_parent_test.values, y_parent_pred.round(), average='macro')