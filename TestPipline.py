import random
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from scipy.spatial import distance
import Cluster_Replacement as CR
import ImageUtil

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


# Higher-level parameter
prototypes_master = []
prototypes_test = []
correct_predictions = [] #[[SamplingCost_i, ClusterReplacement_i]]
num_iterations = 0 # track current iteration of code
num_test_prototypes = 20

DEV_MODE=True # enables definition of image paths in code

# Sampling Cost Parameters
classes = []
prototypes = []
psi = []
error_count = 0
prototypes_master_SC = []
accuracy_perit_SC = []
# Cluster Replacement Parameters
prototype_budget = 8
prototypes_CR= []
prototypes_CR_base= []
prototypes_CR_train= []
required_train_size = 1
max_iterations_CR = 5 # lower to increase performance
accuracy_perit_CR = []

if DEV_MODE:
    img_labeled, img_unlabeled = ImageUtil.get_paths(0,"C:/Users/mason/Desktop/StatMachineLearningProject/Photos/book",30,10)

    temp_labeled, temp_unlabeled = ImageUtil.get_paths(1,"C:/Users/mason/Desktop/StatMachineLearningProject/Photos/car",30,10)
    img_labeled = np.append(img_labeled,temp_labeled,axis=0)
    img_unlabeled = np.append(img_unlabeled,temp_unlabeled,axis=0)

    temp_labeled, temp_unlabeled = ImageUtil.get_paths(2,"C:/Users/mason/Desktop/StatMachineLearningProject/Photos/chair",30, 10)
    img_labeled = np.append(img_labeled, temp_labeled, axis=0)
    img_unlabeled = np.append(img_unlabeled, temp_unlabeled, axis=0)

    temp_labeled, temp_unlabeled = ImageUtil.get_paths(2,"C:/Users/mason/Desktop/StatMachineLearningProject/Photos/kite",30, 10)
    img_labeled = np.append(img_labeled, temp_labeled, axis=0)
    img_unlabeled = np.append(img_unlabeled, temp_unlabeled, axis=0)


    np.random.shuffle(img_labeled)
    np.random.shuffle(img_unlabeled)
    #image_paths = np.append(img_labeled, img_unlabeled, axis=0)

    init_train = img_labeled[:int(0.3 * np.size(img_labeled[:, 0])), :]

    later_train = img_labeled[int(0.3 * np.size(img_labeled[:, 0])):, :]
    later_train= np.append(later_train, img_unlabeled, axis=0)
    np.random.shuffle(later_train)

    image_paths = np.append(init_train,later_train,axis=0)


    #np.random.shuffle(image_paths)


# Sampling Cost Parameters
def pipeline():
    i = True
    #global accuracy_perit_SC,protoypes, error_count, psi, correct_predictions, num_iterations, prototypes_master_SC
    global correct_predictions, num_iterations
    global error_count, psi, accuracy_perit_SC,  prototypes_master_SC
    global prototypes_master, prototypes_CR, prototypes_CR_train, prototypes_CR,accuracy_perit_CR

    while i is True and num_iterations<=np.size(image_paths[:,0]):
        if DEV_MODE: num_iterations+=1
        if DEV_MODE: print('\n\n\n\n#########Iteration: ',num_iterations,'/',np.size(image_paths[:,0]),'#########')
        else: print('\n\n\n\nIteration: ',num_iterations)

        if DEV_MODE: path=image_paths[num_iterations-1,0]
        else: path = input("\n Enter Image Path: " "\n")
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        #print('Feature Shape',np.shape(feature), '\n')

        if DEV_MODE: label_ask=image_paths[num_iterations-1,1]
        else: label_ask = input("Do you want to label image? '\n' Enter N or Y " "\n")

# Testing ############################################################
        if label_ask.lower() == 'n':
            correct_pred = [0,0]

            # Sampling Cost###################################
            predicted_label = get_best_matching_unit(prototypes, feature)
            if DEV_MODE: actual_label = image_paths[num_iterations - 1, 2]
            else: actual_label = input("The actual Class is: ")

            if actual_label.lower() == predicted_label.lower():
                print("\n SC: CORRECT PREDICTION")
                correct_pred[0]=1

            else:
                print("\n SC: ICORRECT PREDICTION")
                error_count += 1
                if error_count > 2:
                    error_count = 0
                    # print("\n Length prototype and psi resp.", len(prototypes), len(psi))
                    prototype_add = SamplingCost(psi)
                    prototypes.append(prototype_add)
                    psi = updateShortTermMemory(psi, prototype_add)
                    print("/n SC: Prototype Added Successfully \n", len(prototypes), len(psi))

            # Cluster Replacement #####################################
            # Format image feature and append class
            flat_feature = to_vector(feature).astype(float)
            P = np.append(flat_feature, actual_label).astype(float)
            # Predict
            predicted_label = CR.predict(P,prototypes_CR)
            #predicted_label = CR.predict(flat_feature, prototypes_CR)
            if predicted_label == actual_label:
                print("CR: CORRECT PREDICTION \n")
                correct_pred[1] = 1
            else: print("CR: INCORRECT PREDICTION \n")

            # Run cluster algorithm on tested prototype
            #prototypes_CR = CR.Cluster_Replacement(prototypes_CR, P, budget=prototype_budget,max_iterations=max_iterations_CR)
            if np.size(prototypes_CR_train) < 1:prototypes_CR_train = np.array([P])  # initialize
            else:prototypes_CR_train = np.append(prototypes_CR_train, np.array([P]), axis=0) # add new prototype to train data
            prototypes_CR = CR.Cluster_Replacement(prototypes_CR, prototypes_CR_train, budget=prototype_budget,
                                                   max_iterations=max_iterations_CR)

            # Update Test Statistics ###################################
            if np.size(correct_predictions) <1: correct_predictions = np.array([correct_pred])
            else: correct_predictions = np.append(correct_predictions, np.array([correct_pred]),axis=0)

# Training #############################################################
        else:
            if DEV_MODE: class_of_image = image_paths[num_iterations - 1, 2]
            else: class_of_image = input("\n Enter the class of this image: " "\n")

            # Sampling Cost###################################
            if class_of_image.lower() not in classes:
                classes.append(class_of_image.lower())
                prototypes.append((feature, class_of_image))
                # print("added feature is {}".format((feature, class_of_image)))

            else:
                d1, d2 = w_updates(class_of_image, feature)
                if len(psi) > 200:
                    psi.pop(0)
                    psi.append((feature, class_of_image, d1, d2))
                else:
                    psi.append((feature, class_of_image, d1, d2))


            # Cluster Replacement #####################################
            # Format image feature and append class
            flat_feature = to_vector(feature).astype(float)
            P = np.append(flat_feature, class_of_image).astype(float)

            # Generate the base codebook for Cluster replacement
            if np.size(prototypes_CR) < 1:
                prototypes_CR = np.array([P])  # initialize
            elif np.size(prototypes_CR[:, 0]) < prototype_budget:  # =required_train_size: #prototype_budget/2.0:
                # If insuficent number of initial prototypes then add raw feature & class to CR prototypes
                print('Add more images before cluster replacment can start..')
                prototypes_CR = np.append(prototypes_CR, np.array([P]), axis=0)
            else:
                if np.shape(prototypes_CR_train)[0] > required_train_size:  # prototype_budget/2:
                    prototypes_CR = CR.Cluster_Replacement(prototypes_CR, prototypes_CR_train,
                                                           budget=prototype_budget, max_iterations=max_iterations_CR)
                    # prototypes_CR_train = [] # reset training prototypes
                else:
                    # Batching
                    if np.size(prototypes_CR_train) < 1:
                        prototypes_CR_train = np.array([P])  # initialize
                    else:
                        prototypes_CR_train = np.append(prototypes_CR_train, np.array([P]),
                                                        axis=0)  # add new prototype to train data




# Update Master Variables ##############
        # Initialize or append master prototype list
        if np.size(prototypes_master) < 1: prototypes_master = np.array([P])
        else: prototypes_master = np.append(prototypes_master, np.array([P]), axis=0)

        prototypes_master_SC.append((feature, class_of_image))

        # if num_test_prototypes<np.size(prototypes_master[:,0]):
        #     prototypes_master=np.delete(prototypes_master,0,0)
        #     del prototypes_master_SC[0]

# Report Statistics #################
        print("\n ## ITERATION ",num_iterations," REPORT ##")
        print("Provided Label: ",image_paths[num_iterations-1,1])
        print("")

        print("Master Codebook Shape: ", np.shape(prototypes_master))
        print("")

        #print("CR Trainable: ",np.size(prototypes_CR[:,0])>=required_train_size)
        #print("CR Base Shape: ", np.shape(prototypes_CR))
        print("CR Train Shape: ", np.shape(prototypes_CR_train))
        print("CR Codebook Shape: ",np.shape(prototypes_CR))
        print("")

        print('Predictions [SC,CR]: ',correct_predictions)
        if np.size(correct_predictions)>1:
            SC_acc = np.mean(correct_predictions[:,0])
            CR_acc = np.mean(correct_predictions[:, 1])
            print("SC Rolling Acc: ", SC_acc)
            print("CR Rolling Acc: ", CR_acc)
        else:
            print("Sampling Cost Rolling Acc: ", "N/A")
            print("Cluster Replacement Rolling Acc: ", "N/A")

        SC_acc = SC_accuracy(prototypes,prototypes_master_SC)
        CR_acc = CR.current_accuracy(prototypes_CR,prototypes_master)
        accuracy_perit_SC.append(SC_acc)
        accuracy_perit_CR.append(CR_acc)
        print("SC Acc: ", SC_acc)
        print("CR Acc: ", CR_acc)

        print("\n############## end ############## ")
        if num_iterations>=np.size(image_paths[:,0])-2:
            print('predictions saved...')
            np.save('predictions', correct_predictions)
            t=np.arange(len( accuracy_perit_SC))
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(accuracy_perit_SC)
            ax.plot(accuracy_perit_CR)
            ax.legend(['Sampling Cost', 'Cluster Replacement'])

            ax.set(xlabel='Iteration', ylabel='Accuracy',
                   title='Sampling Cost vs Cluster Replacement Accuracy per Iteration')
           #ax.grid()

            fig.savefig("accuracyfig.png")
            plt.show()

def mu(d1, d2):
    mu_ = (np.subtract(d1, d2))/(np.add(d1, d2))
    return mu_


def deriv(mu_, t=2):
    z = (t*(np.exp(-t*mu_)))/((1 + np.exp(-t*mu_))**2)
    return z


def get_best_matching_unit(prototype_list, sample):
    distances = []
    for prototype in prototype_list:
        dist = squared_euclidean(prototype[0], sample)
        distances.append((prototype, dist))
    # print(distances)
    distances.sort(key=lambda tup: tup[1])
    # print("\n\n", distances, type(distances), distances[0], type(distances[0]))
    #print("Predicted Class: \n", distances[0][0][1])
    return distances[0][0][1]


def w_updates(class_of_image, feature, alpha=0.01):
    current_w1 = 0
    current_w2 = 0
    index_w1 = 0
    index_w2 = 0

    lowest_distance_same_class = float('inf')
    for i in range(len(prototypes)):
        if class_of_image in prototypes[i]:
            # print(prototypes[i])
            temp_lowest_same_class = squared_euclidean(feature, prototypes[i][0])
            # print("templowestsamplesame is {}".format(temp_lowest_same_class))
            if temp_lowest_same_class <= lowest_distance_same_class:
                lowest_distance_same_class = temp_lowest_same_class
                current_w1 = prototypes[i][0]
                index_w1 = i
                # print("\n Index w1 {} ".format(index_w1))

    d1 = lowest_distance_same_class

    list_w2 = [item for item in prototypes if class_of_image not in item]
    # print(len(list_w2), list_w2)
    if len(list_w2) == 0:
        d1 = 0
        d2 = 0
        return d1, d2

    lowest_distance_different_class = float('inf')
    for j in range(len(prototypes)):
        if class_of_image not in prototypes[j]:
            # print(prototypes[j])
            temp_lowest_different_class = squared_euclidean(feature, prototypes[j][0])
            # print("templowestsampledifferent is {}".format(temp_lowest_different_class))
            if temp_lowest_different_class <= lowest_distance_different_class:
                lowest_distance_different_class = temp_lowest_different_class
                current_w1 = prototypes[j][0]
                index_w2 = j
                # print("\n Index w2 {} ".format(index_w2))

    d2 = lowest_distance_different_class

    mu_ = mu(d1, d2)
    z = deriv(mu_)

    w1 = current_w1 + (alpha*z*4*d2*(feature - current_w1))/((np.add(d1, d2))**2)
    w2 = current_w2 - (alpha*z*4*d1*(feature - current_w2))/((np.add(d1, d2))**2)
    class_w1 = prototypes[index_w1][1]
    # print("class w1 is {}".format(class_w1))
    class_w2 = prototypes[index_w2][1]
    # print("class w2 is {}".format(class_w2))
    prototypes[index_w1] = (w1, class_w1)
    prototypes[index_w2] = (w2, class_w2)
    return d1, d2


def SamplingCost(psi, num_samples=10):
    print("\n Running SamplingCost\n")
    # print("\n Len psi is {}".format(len(psi)), '\n')
    if len(psi) < num_samples:
        num_samples = len(psi)

    # print(num_samples, '\n')
    minCost = float('inf')
    new_prototype = None
    psi_random = random.sample(psi, num_samples)
    # print("Random samples are \n {}".format(psi_random), len(psi_random))
    for ran_sam in range(num_samples):
        # print("\n RANSAM \n", ran_sam)
        (x, y) = (psi_random[ran_sam][0], psi_random[ran_sam][1])
        # print("XY is {} \n ".format((x, y)))
        psi_temp = updateShortTermMemory(psi, (x, y))
        calc_cost = calculateCost(psi_temp)
        if calc_cost < minCost:
            minCost = calc_cost
            new_prototype = (x, y)

    return new_prototype


def calculateCost(psi_temp):
    print("\n Calculating cost \n")
    cost = 0
    for i in range(len(psi_temp)):
        temp_mu = mu(psi_temp[i][2], psi_temp[i][3])
        temp_deriv = deriv(temp_mu)
        cost += temp_deriv
    return cost


def updateShortTermMemory(psi, x_y):
    print("\n Updating Short term memory \n")
    psi_temp = psi
    for iter_d in range(len(psi)):
        if psi_temp[iter_d][1] == x_y[1]:
            d_plus = squared_euclidean(psi_temp[iter_d][0], x_y[0])
            if psi_temp[iter_d][2] > d_plus:
                temp_x, temp_y, temp_dminus = psi_temp[iter_d][0], psi_temp[iter_d][1], psi_temp[iter_d][3]
                psi_temp[iter_d] = (temp_x, temp_y, d_plus, temp_dminus)

        if psi_temp[iter_d][1] != x_y[1]:
            # print("\n Going in is {} and {} \n".format(psi_temp[iter_d][0], x_y[0]))
            d_minus = squared_euclidean(psi_temp[iter_d][0], x_y[0])
            if psi_temp[iter_d][3] > d_minus:
                temp_x, temp_y, temp_dplus = psi_temp[iter_d][0], psi_temp[iter_d][1], psi_temp[iter_d][2]
                psi_temp[iter_d] = (temp_x, temp_y, temp_dplus, d_minus)

    return psi_temp

def to_vector(prototype_tuple):
    P = np.ndarray.flatten(prototype_tuple)
    return P

def squared_euclidean(x, w):
    x = np.ndarray.flatten(x)
    w = np.ndarray.flatten(w)
    dst = distance.euclidean(x, w)
    return dst**2

def SC_accuracy(current_prototypes,prototypes_test):
    preds = []
    for P in prototypes_test:
        feature = P[0]
        class_of_image = P[1]
        predicted_label = get_best_matching_unit(prototypes, feature)
        if class_of_image==predicted_label: preds.append(1)
        else: preds.append(0)
    return np.mean(preds)

pipeline()