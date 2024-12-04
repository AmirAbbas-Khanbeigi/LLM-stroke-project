import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#import seaborn as sns
from scipy.io import loadmat
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from sklearn.manifold import TSNE
import scipy.stats
import umap
from sklearn.metrics import silhouette_score
import random

import importlib
#import VisFunctions

#importlib.reload(VisFunctions)
#from VisFunctions import Rain_Cloud_vis


import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import LocalOutlierFactor


from scipy.stats import mannwhitneyu
import skdim


def shuffle_data_and_labels_strings(data, labels, seed=None):
    # Pair up the data and labels to maintain correspondence
    
    combined = list(zip(data, labels))
    
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle the combined list
    np.random.shuffle(combined)
    
    # Unzip the shuffled combined list back into data and labels
    shuffled_data, shuffled_labels = zip(*combined)
    
    return list(shuffled_data), list(shuffled_labels)

                               

def bootstrapper_subject_consistent(controls_index_list,
                      patients_index_list,
                      vectors_array_6FC,
                      labels_array_6FC,
                      control_vs_patient_labels,
                      bootstrap_seed,
                      bootstrap_fraction, #= 0.8,
                      N_datapoint_of_subject
                      ):
    
    #working: 133 (2/22), 135 (4/22), 141 (6/32)
    #not working:239, 134, 136, 137, 138, 139, 140
    controls_num_zeros = int(len(controls_index_list)*(1-bootstrap_fraction))
    controls_sample_size = len(controls_index_list) - controls_num_zeros
    controls_selection_arrayy = np.array([0] * controls_num_zeros + [1] * controls_sample_size)
    np.random.seed(bootstrap_seed)  # Setting the seed for reproducibility
    np.random.shuffle(controls_selection_arrayy)
    #print('controls_selection_arrayy= ', controls_selection_arrayy)

    subsample_controls_index= []
    for i,value in enumerate(controls_selection_arrayy):
        if value==1:
            subsample_controls_index.append(controls_index_list[i])
    #print('subsample_controls_index= ', subsample_controls_index)
        

    patients_num_zeros = int(len(patients_index_list)*(1-bootstrap_fraction))
    patients_sample_size = len(patients_index_list) - patients_num_zeros
    patients_selection_arrayy = np.array([0] * patients_num_zeros + [1] * patients_sample_size)
    np.random.seed(bootstrap_seed)  # Setting the seed for reproducibility
    np.random.shuffle(patients_selection_arrayy)
    #print('patients_selection_arrayy= ', patients_selection_arrayy)

    subsample_patients_index= []
    for i,value in enumerate(patients_selection_arrayy):
        if value==1:
            subsample_patients_index.append(patients_index_list[i])
    #print('subsample_patients_index= ', subsample_patients_index)

    controls_patients_selection_arrayy= np.concatenate([controls_selection_arrayy , patients_selection_arrayy])

    #to select the consistently so that all of 6 FCs of a subject are selected together
    data_subsample= []
    label_6FC_subsample= []
    control_vs_patient_labels_subsample= []
    for i,value in enumerate(controls_patients_selection_arrayy):
        if value==1:
            for j in range(N_datapoint_of_subject):
                data_subsample.append(vectors_array_6FC[i*N_datapoint_of_subject+j])
                label_6FC_subsample.append(labels_array_6FC[i*N_datapoint_of_subject+j])
                control_vs_patient_labels_subsample.append(control_vs_patient_labels[i*N_datapoint_of_subject+j])

    data_subsample= np.array(data_subsample)
    label_6FC_subsample= np.array(label_6FC_subsample)
    control_vs_patient_labels_subsample= np.array(control_vs_patient_labels_subsample)
    # print('data_subsample.shape= ', data_subsample.shape)
    # print('label_6FC_subsample= ', label_6FC_subsample)
    # print('controls_patients_6FC_labels= ', controls_patients_6FC_labels)
    return  (subsample_controls_index,
             subsample_patients_index,
             data_subsample,
             label_6FC_subsample,
             control_vs_patient_labels_subsample
             )





# Example usage:
# Data = [[1, 2], [3, 4], [5, 6]]
# Labels = ['first', 'second', 'third']

# shuffled_Data, shuffled_Labels = shuffle_data_and_labels_strings(Data, Labels,)
# print("Shuffled Data:", shuffled_Data)
# print("Shuffled Labels:", shuffled_Labels)




def dictionary_random_reducer(data, seed, N):
  
    dic_length= len(list(data.values()))
    if N > dic_length:
        raise ValueError("N must be less than or equal to the number of items in the dictionary")

    random.seed(seed)
    lst = [1] * N + [0] * (dic_length - N)
    random.shuffle(lst)

    dicc= {}
    for i,key in enumerate(data.keys()):
        if lst[i]==1:
            dicc[key]= data[key]

    return dicc

# # Example usage
# my_dict = {f'{i}': i*2 for i in range(100)}
# print(my_dict)
# seed = 42
# N = 25

# reduced_dic = dictionary_random_reducer(my_dict, seed, N)
# print(reduced_dic)

def UMAP_implementation(seed_list, #number of shufflings
                        gamma_list,
                        alpha_list,
                        metricc_list,
                        n_neighbor_list,
                        min_dist_list,
                        epochs_list,
                        concatenated_data,
                        D_Emb, #dimension of embedding
                        N_B, # number of bootstrappings  
                        B_index): # bootstrappings index

    dataa= np.zeros_like(concatenated_data)
    label_like_array= np.zeros(len(concatenated_data))
    umap_embed_vs_params_N_shuffled= np.zeros((len(seed_list),
                                    len(gamma_list),
                                    len(alpha_list),
                                    len(metricc_list),
                                    len(n_neighbor_list),
                                    len(min_dist_list),
                                    len(epochs_list),
                                    len(concatenated_data),
                                    D_Emb))

    number_of_plots= len(n_neighbor_list)*len(min_dist_list)*len(metricc_list)*len(gamma_list)*len(alpha_list)*len(epochs_list)*len(seed_list)*N_B
    counter= (number_of_plots/N_B)*B_index
    for n,seedd in enumerate(seed_list): 
        for m,gamma in enumerate(gamma_list): 
            for l,alpha in enumerate(alpha_list): 
                for k,metricc in enumerate(metricc_list): 
                    for i,n_neighb in enumerate(n_neighbor_list): 
                        for j,min_dist in enumerate(min_dist_list):
                            for o,epoch in enumerate(epochs_list):
                                print("#: ", counter, " out of ", number_of_plots)
                                # print(n, "-th shuffling")
                                print("n_neighbor= ", n_neighb)
                                # print("min_dist= ", min_dist)
                                # print("repulsion= ", gamma)
                                # print("learning rate= ", alpha)
                                print("metric= ", metricc)

                                umap_setting = umap.UMAP(
                                            n_components= D_Emb,
                                            n_neighbors=int(n_neighb),
                                            min_dist=min_dist,
                                            metric=metricc,
                                            random_state=43,
                                            repulsion_strength= gamma,
                                            learning_rate= alpha#,
                                            #n_epochs= epoch
                                )

                                time1= datetime.now().strftime("%H:%M:%S")
                                time1 = datetime.strptime(time1, "%H:%M:%S")    
                                
                                
        #                         if n==0:
        #                             dataa = random_array
        #                         if n==1:
                                dataa , _ = shuffle_data_and_labels_strings(concatenated_data,
                                                    label_like_array, seedd)
                                    
                                umap_embed_vs_params_N_shuffled[n, m, l, k, i ,j, o]= umap_setting.fit_transform(dataa)
                                #print(umap_embed_vs_params_N_shuffled[n, m, l, k, i ,j, o])
                                #print("Time:", datetime.now().strftime("%H:%M:%S"))
                                #umap_embed_vs_n_neighbor[k] = reducer.fit_transform(concatenated_vec_Schaeffer_300_01003_RS_Run1)


                                time2= datetime.now().strftime("%H:%M:%S")
                                time2 = datetime.strptime(time2, "%H:%M:%S")
                                time_spent= time2 - time1
                                print("calculation time= ", time_spent.total_seconds(), "\n")
                                counter+= 1
    return  umap_embed_vs_params_N_shuffled

# to see how linear they are scattered

from scipy.stats import linregress
from sklearn.decomposition import PCA


def perpendicular_distances(x, y, slope, intercept):
    # For line ax + by + c = 0, here a = -slope, b = 1, c = -intercept
    a, b, c = -slope, 1, -intercept
    distances = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
    return np.array(distances)

# x=np.array([1,2,3,4,5,6])
# y=np.array([1,3,-4,2,1,0])


def fit_line_and_calculate_distances(points):
    pca = PCA(n_components=1)
    pca.fit(points)
    line_direction = pca.components_[0]
    
    # Project points onto the PCA line to find the closest points on the line
    points_projected = pca.transform(points)
    points_projected_reconstructed = pca.inverse_transform(points_projected)
    
    # Calculate perpendicular distances
    distances = np.linalg.norm(points - points_projected_reconstructed, axis=1)
    sum_distances = np.sum(distances)
    return sum_distances

def stat_measure_low_dim_linearity(seed_list, #number of shufflings
                        bootstrap_seed_list,
                        bootstrap_fraction,
                        gamma_list,
                        alpha_list,
                        metricc_list,
                        n_neighbor_list,
                        min_dist_list,
                        epochs_list,
                        umap_embed_vs_params_N_shuffled,
                        controls_patients_6FC_labels,
                        controls_index_list,
                        patients_index_list,
                        D_Emb,
                        Rain_Cloud_show_bool): #dimension of embedding

    # counter=0
    # significant_correlations_list= []

    # all_t_test_p_values_nonshuffled= []
    # all_t_test_p_values_shuffled1= []

    # #non_shuffled
    # significant_t_test_list_nonshuffled= []
    # t_test_p_values_nonshuffled= []
    # t_test_p_values_and_params_nonshuffled= [] #gamma alpha metrixx n_neighbor min_dist
    # sigma_difference_nonshuffled= []

    # #shuffled 1
    # significant_t_test_list_shuffled1= []
    # t_test_p_values_shuffled1= []
    # t_test_p_values_and_params_shuffled1= [] #gamma alpha metrixx n_neighbor min_dist
    # sigma_difference_shuffled1= []


    # repulsion_significant_list= []
    # repulsion_nonsignificant_list= []

    # alpha_significant_list= []
    # alpha_nonsignificant_list= []

    # n_neighbor_significant_list= []
    # n_neighbor_nonsignificant_list= []
    #################################

    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_2D = []
    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D= [] #list of dictionaries
    t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D= [] #list of dictionaries

    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_3D = []
    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D= [] #list of dictionaries
    t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D= [] #list of dictionaries

    # measure_stat_params_Avg_shuff= np.zeros((len(gamma_list),
    #                                     len(alpha_list),
    #                                     len(metricc_list),
    #                                     len(n_neighbor_list),
    #                                     len(min_dist_list),
    #                                     len(epochs_list)
    #                                     ))

    for m,gamma in enumerate(gamma_list): 
        for l,alpha in enumerate(alpha_list): 
            for k,metricc in enumerate(metricc_list): 
                for i,n_neighb in enumerate(n_neighbor_list): 
                    for j,min_dist in enumerate(min_dist_list):
                        for o,epoch in enumerate(epochs_list):
                            p_values_of_N_boots_Avg_shuff_2D= []
                            p_values_of_N_boots_Avg_shuff_3D= []
                            for p, B_seed in enumerate(bootstrap_seed_list):
                                p_values_of_N_shuff_2D= []
                                p_values_of_N_shuff_3D= []
                                for n,seedd in enumerate(seed_list):
                                    #counter+= 1
                                    
                                    (subsample_controls_index,
                                    subsample_patients_index,
                                    _,
                                    label_6FC_subsample,
                                    _)= bootstrapper_subject_consistent(controls_index_list,
                                                                        patients_index_list,
                                                                        np.zeros_like(controls_patients_6FC_labels),
                                                                        controls_patients_6FC_labels,
                                                                        np.zeros_like(controls_patients_6FC_labels),
                                                                        B_seed,
                                                                        bootstrap_fraction,#= 0.8,
                                                                        6) # N_datapoint_of_subject
                                    
                                    _ , shuffled_string_lables = shuffle_data_and_labels_strings(np.zeros_like(label_6FC_subsample),
                                                        label_6FC_subsample, seedd)   
                                
                                    umap_latent= umap_embed_vs_params_N_shuffled[p, n, m, l, k, i ,j, o]

                                    #####

                                    if D_Emb==2:
                                        controls_line_proj_dis_2D= np.zeros(len(controls_index_list))
                                        patients_line_proj_dis_2D= np.zeros(len(patients_index_list))
                                    if D_Emb==3:
                                        controls_line_proj_dis_3D= np.zeros(len(controls_index_list))
                                        patients_line_proj_dis_3D= np.zeros(len(patients_index_list))


                                    #xx= np.zeros(len(controls_index_list)+len(patients_index_list))
                                    #yy= np.zeros(len(controls_index_list)+len(patients_index_list))
                                    
                                    if D_Emb==2:
                                        x_range= np.max(umap_latent[:,0])- np.min(umap_latent[:,0])
                                        y_range= np.max(umap_latent[:,1])- np.min(umap_latent[:,1])
                                    if D_Emb==3:
                                        x_range= np.max(umap_latent[:,0])- np.min(umap_latent[:,0])
                                        y_range= np.max(umap_latent[:,1])- np.min(umap_latent[:,1])                           
                                        z_range= np.max(umap_latent[:,2])- np.min(umap_latent[:,2])                           
                                    

                                    #print("n= ", n)
                                    for ii, ID in enumerate(subsample_controls_index):
                                        #print(f"control {ID}: ")
                                        xxx= [] #np.zeros(6)
                                        yyy= [] #np.zeros(6)
                                        if D_Emb==3:
                                            zzz=[]
                                        for jj,label in enumerate(shuffled_string_lables):
                                            if (label[0]=='c'and label[1:3]==f"{ID:02d}"):
                                                #print(umap_latent[jj])
                                                xxx.append(umap_latent[jj,0])
                                                yyy.append(umap_latent[jj,1])
                                                if D_Emb==3:
                                                    zzz.append(umap_latent[jj,2])
                                                    #print("")
                                                #print(label)

                                        xxx= np.array(xxx)
                                        yyy= np.array(yyy)
                                        if D_Emb==3:
                                            zzz= np.array(zzz)
                                        
                                        #print("xxx= ", xxx)
                                        #print("yyy= ", yyy)
                                        if D_Emb==2:
                                            slope, intercept, _, _, _ = linregress(xxx, yyy)
                                            controls_line_proj_dis_2D[ii]= np.sum(perpendicular_distances(xxx, yyy, slope, intercept))
                                            #print('n_neighb= ',n_neighb)
                                            #print("sum of perpend distances= ", np.sum(perpendicular_distances(xxx, yyy, slope, intercept)))

                                        if D_Emb==3:
                                            points= [[xxx[i],yyy[i],zzz[i]] for i in range(len(xxx))]
                                            points= np.array(points)
                                            controls_line_proj_dis_3D[ii]= fit_line_and_calculate_distances(points)
                                        
                                    for ii, ID in enumerate(subsample_patients_index):
                                        #print(f"patient {ID}: ")
                                        xxx= [] #np.zeros(6)
                                        yyy= [] #np.zeros(6)
                                        if D_Emb==3:
                                            zzz=[]
                                        for jj,label in enumerate(shuffled_string_lables):
                                            if (label[0]=='p'and label[1:3]==f"{ID:02d}"):
                                                #print(umap_latent[jj])
                                                xxx.append(umap_latent[jj,0])
                                                yyy.append(umap_latent[jj,1])
                                                if D_Emb==3:
                                                    zzz.append(umap_latent[jj,2])
                                                #print(label)

                                        xxx= np.array(xxx)
                                        yyy= np.array(yyy)
                                        if D_Emb==3:
                                            zzz= np.array(zzz)
                                        
                                        #print("xxx= ", xxx)
                                        #print("yyy= ", yyy)

                                        if D_Emb==2:
                                            slope, intercept, _, _, _ = linregress(xxx, yyy)
                                            patients_line_proj_dis_2D[ii]= np.sum(perpendicular_distances(xxx, yyy, slope, intercept))
                                            #print('n_neighb= ',n_neighb)
                                            #print("sum of perpend distances= ", np.sum(perpendicular_distances(xxx, yyy, slope, intercept)))


                                        if D_Emb==3:
                                            points= [[xxx[i],yyy[i],zzz[i]] for i in range(len(xxx))]
                                            points= np.array(points)
                                            patients_line_proj_dis_3D[ii]= fit_line_and_calculate_distances(points)                
                        
                                    if D_Emb==2:
                                        #print("\n","@@@@@@@@@@@@@@@@@@@@@", "\n")
                                        print('n_neighb= ',n_neighb)
                                        #print('controls_line_proj_dis_2D= ',controls_line_proj_dis_2D)
                                        #print('patients_line_proj_dis_2D= ',patients_line_proj_dis_2D)
                                        t_stat_2D, t_stat_p_value_2D = scipy.stats.ttest_ind(controls_line_proj_dis_2D, patients_line_proj_dis_2D)
                                        p_values_of_N_shuff_2D.append(t_stat_p_value_2D)
                                        #print("p-value (2D; one realization)= ", t_stat_p_value_2D)
                                        if Rain_Cloud_show_bool==True:
                                            Rain_Cloud_vis(controls_line_proj_dis_2D, patients_line_proj_dis_2D, 'controls', 'patients')
                                        if t_stat_p_value_2D < 0.05:
                                            print("p-value (2D; one realization)= ", t_stat_p_value_2D)
                                        #     if Rain_Cloud_show_bool==True:
                                        #         Rain_Cloud_vis(controls_line_proj_dis_2D, patients_line_proj_dis_2D, 'controls', 'patients')
                                    
                                    if D_Emb==3:
                                        t_stat_3D, t_stat_p_value_3D = scipy.stats.ttest_ind(controls_line_proj_dis_3D, patients_line_proj_dis_3D)
                                        p_values_of_N_shuff_3D.append(t_stat_p_value_3D)
                                        if t_stat_p_value_3D < 0.05:
                                            print("p-value (3D; one realization)= ", t_stat_p_value_3D)
                                            if Rain_Cloud_show_bool==True:
                                                Rain_Cloud_vis(controls_line_proj_dis_3D, patients_line_proj_dis_3D, 'controls', 'patients')

            #                         all_t_test_p_values_shuffled1.append(t_stat_p_value)
            #                         if t_stat_p_value > 0.05:
            #                             repulsion_nonsignificant_list.append(gamma)
            #                             alpha_nonsignificant_list.append(alpha)
            #                             n_neighbor_nonsignificant_list.append(n_neighb)
                                
                                if D_Emb==2:
                                    p_values_of_N_shuff_2D= np.array(p_values_of_N_shuff_2D)
                                    #print("Avg p-value (2D)= ", np.mean(p_values_of_N_shuff_2D))
                                    p_values_of_N_boots_Avg_shuff_2D.append(np.mean(p_values_of_N_shuff_2D))
                                if D_Emb==3:
                                    p_values_of_N_shuff_3D= np.array(p_values_of_N_shuff_3D)
                                    #print("Avg p-value (3D)= ", np.mean(p_values_of_N_shuff_3D))
                                    p_values_of_N_boots_Avg_shuff_3D.append(np.mean(p_values_of_N_shuff_3D))

                            if D_Emb==2:
                                p_values_of_N_boots_Avg_shuff_2D= np.array(p_values_of_N_boots_Avg_shuff_2D)
                                #print("p-value for Avg boots Avg shuff= ", np.mean(p_values_of_N_boots_Avg_shuff_2D))
                                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_2D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})        
                                if np.mean(p_values_of_N_boots_Avg_shuff_2D) < 0.05:
                                    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_2D.append(np.mean(p_values_of_N_boots_Avg_shuff_2D))
                                    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_2D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})

                            if D_Emb==3:
                                p_values_of_N_boots_Avg_shuff_3D= np.array(p_values_of_N_boots_Avg_shuff_3D)
                                print("p-value for Avg boots Avg shuff= ", np.mean(p_values_of_N_boots_Avg_shuff_3D))
                                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_3D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})        
                                if np.mean(p_values_of_N_boots_Avg_shuff_3D) < 0.05:
                                    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_3D.append(np.mean(p_values_of_N_boots_Avg_shuff_3D))
                                    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_3D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})
    if D_Emb==2:
        return (signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_2D,
                signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D, #list of dictionaries
                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D) #list of dictionaries
    if D_Emb==3:        
        return (signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_3D,
                signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D, #list of dictionaries
                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D) #list of dictionaries


def TS_stat_measure_low_dim_linearity(seed_list, #number of shufflings
                        bootstrap_seed_list,
                        bootstrap_fraction,
                        gamma_list,
                        alpha_list,
                        metricc_list,
                        n_neighbor_list,
                        min_dist_list,
                        epochs_list,
                        umap_embed_vs_params_N_shuffled,
                        controls_patients_TS_labels,
                        controls_index_list,
                        patients_index_list,
                        D_Emb,
                        N_datapoint_of_subject,
                        Rain_Cloud_show_bool): #dimension of embedding

    # counter=0
    # significant_correlations_list= []

    # all_t_test_p_values_nonshuffled= []
    # all_t_test_p_values_shuffled1= []

    # #non_shuffled
    # significant_t_test_list_nonshuffled= []
    # t_test_p_values_nonshuffled= []
    # t_test_p_values_and_params_nonshuffled= [] #gamma alpha metrixx n_neighbor min_dist
    # sigma_difference_nonshuffled= []

    # #shuffled 1
    # significant_t_test_list_shuffled1= []
    # t_test_p_values_shuffled1= []
    # t_test_p_values_and_params_shuffled1= [] #gamma alpha metrixx n_neighbor min_dist
    # sigma_difference_shuffled1= []


    # repulsion_significant_list= []
    # repulsion_nonsignificant_list= []

    # alpha_significant_list= []
    # alpha_nonsignificant_list= []

    # n_neighbor_significant_list= []
    # n_neighbor_nonsignificant_list= []
    #################################

    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_2D = []
    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D= [] #list of dictionaries
    t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D= [] #list of dictionaries

    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_3D = []
    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D= [] #list of dictionaries
    t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D= [] #list of dictionaries

    # measure_stat_params_Avg_shuff= np.zeros((len(gamma_list),
    #                                     len(alpha_list),
    #                                     len(metricc_list),
    #                                     len(n_neighbor_list),
    #                                     len(min_dist_list),
    #                                     len(epochs_list)
    #                                     ))

    for m,gamma in enumerate(gamma_list): 
        for l,alpha in enumerate(alpha_list): 
            for k,metricc in enumerate(metricc_list): 
                for i,n_neighb in enumerate(n_neighbor_list): 
                    for j,min_dist in enumerate(min_dist_list):
                        for o,epoch in enumerate(epochs_list):
                            p_values_of_N_boots_Avg_shuff_2D= []
                            p_values_of_N_boots_Avg_shuff_3D= []
                            for p, B_seed in enumerate(bootstrap_seed_list):
                                p_values_of_N_shuff_2D= []
                                p_values_of_N_shuff_3D= []
                                for n,seedd in enumerate(seed_list):
                                    #counter+= 1
                                    
                                    (subsample_controls_index,
                                    subsample_patients_index,
                                    _,
                                    label_TS_subsample,
                                    _)= bootstrapper_subject_consistent(controls_index_list,
                                                                        patients_index_list,
                                                                        np.zeros_like(controls_patients_TS_labels),
                                                                        controls_patients_TS_labels,
                                                                        np.zeros_like(controls_patients_TS_labels),
                                                                        B_seed,
                                                                        bootstrap_fraction,#= 0.8 bootstrap_fraction
                                                                        N_datapoint_of_subject) # N_datapoint_of_subject # 72 or 144
                                

                                    _ , shuffled_string_lables = shuffle_data_and_labels_strings(np.zeros_like(label_TS_subsample),
                                                        label_TS_subsample, seedd)   
                                
                                    umap_latent= umap_embed_vs_params_N_shuffled[p, n, m, l, k, i ,j, o]

                                    #####

                                    if D_Emb==2:
                                        controls_line_proj_dis_2D= np.zeros(len(controls_index_list))
                                        patients_line_proj_dis_2D= np.zeros(len(patients_index_list))
                                    if D_Emb==3:
                                        controls_line_proj_dis_3D= np.zeros(len(controls_index_list))
                                        patients_line_proj_dis_3D= np.zeros(len(patients_index_list))


                                    #xx= np.zeros(len(controls_index_list)+len(patients_index_list))
                                    #yy= np.zeros(len(controls_index_list)+len(patients_index_list))
                                    
                                    if D_Emb==2:
                                        x_range= np.max(umap_latent[:,0])- np.min(umap_latent[:,0])
                                        y_range= np.max(umap_latent[:,1])- np.min(umap_latent[:,1])
                                    if D_Emb==3:
                                        x_range= np.max(umap_latent[:,0])- np.min(umap_latent[:,0])
                                        y_range= np.max(umap_latent[:,1])- np.min(umap_latent[:,1])                           
                                        z_range= np.max(umap_latent[:,2])- np.min(umap_latent[:,2])                           
                                    

                                    #print("n= ", n)
                                    for ii, ID in enumerate(subsample_controls_index):
                                        #print(f"control {ID}: ")
                                        xxx= [] #np.zeros(6)
                                        yyy= [] #np.zeros(6)
                                        if D_Emb==3:
                                            zzz=[]
                                        for jj,label in enumerate(shuffled_string_lables):
                                            if (label[0]=='c'and label[1:3]==f"{ID:02d}"):
                                                #print(umap_latent[jj])
                                                xxx.append(umap_latent[jj,0])
                                                yyy.append(umap_latent[jj,1])
                                                if D_Emb==3:
                                                    zzz.append(umap_latent[jj,2])
                                                    #print("")
                                                #print(label)

                                        xxx= np.array(xxx)
                                        yyy= np.array(yyy)
                                        if D_Emb==3:
                                            zzz= np.array(zzz)
                                        
                                        #print("xxx= ", xxx)
                                        #print("yyy= ", yyy)
                                        if D_Emb==2:
                                            slope, intercept, _, _, _ = linregress(xxx, yyy)
                                            controls_line_proj_dis_2D[ii]= np.sum(perpendicular_distances(xxx, yyy, slope, intercept))
                                            #print('n_neighb= ',n_neighb)
                                            #print("sum of perpend distances= ", np.sum(perpendicular_distances(xxx, yyy, slope, intercept)))

                                        if D_Emb==3:
                                            points= [[xxx[i],yyy[i],zzz[i]] for i in range(len(xxx))]
                                            points= np.array(points)
                                            controls_line_proj_dis_3D[ii]= fit_line_and_calculate_distances(points)
                                        
                                    for ii, ID in enumerate(subsample_patients_index):
                                        #print(f"patient {ID}: ")
                                        xxx= [] #np.zeros(6)
                                        yyy= [] #np.zeros(6)
                                        if D_Emb==3:
                                            zzz=[]
                                        for jj,label in enumerate(shuffled_string_lables):
                                            if (label[0]=='p'and label[1:3]==f"{ID:02d}"):
                                                #print(umap_latent[jj])
                                                xxx.append(umap_latent[jj,0])
                                                yyy.append(umap_latent[jj,1])
                                                if D_Emb==3:
                                                    zzz.append(umap_latent[jj,2])
                                                #print(label)

                                        xxx= np.array(xxx)
                                        yyy= np.array(yyy)
                                        if D_Emb==3:
                                            zzz= np.array(zzz)
                                        
                                        #print("xxx= ", xxx)
                                        #print("yyy= ", yyy)

                                        if D_Emb==2:
                                            slope, intercept, _, _, _ = linregress(xxx, yyy)
                                            patients_line_proj_dis_2D[ii]= np.sum(perpendicular_distances(xxx, yyy, slope, intercept))
                                            #print('n_neighb= ',n_neighb)
                                            #print("sum of perpend distances= ", np.sum(perpendicular_distances(xxx, yyy, slope, intercept)))


                                        if D_Emb==3:
                                            points= [[xxx[i],yyy[i],zzz[i]] for i in range(len(xxx))]
                                            points= np.array(points)
                                            patients_line_proj_dis_3D[ii]= fit_line_and_calculate_distances(points)                
                        
                                    if D_Emb==2:
                                        #print("\n","@@@@@@@@@@@@@@@@@@@@@", "\n")
                                        print('n_neighb= ',n_neighb)
                                        #print('controls_line_proj_dis_2D= ',controls_line_proj_dis_2D)
                                        #print('patients_line_proj_dis_2D= ',patients_line_proj_dis_2D)
                                        t_stat_2D, t_stat_p_value_2D = scipy.stats.ttest_ind(controls_line_proj_dis_2D, patients_line_proj_dis_2D)
                                        p_values_of_N_shuff_2D.append(t_stat_p_value_2D)
                                        #print("p-value (2D; one realization)= ", t_stat_p_value_2D)
                                        if Rain_Cloud_show_bool==True:
                                            Rain_Cloud_vis(controls_line_proj_dis_2D, patients_line_proj_dis_2D, 'controls', 'patients')
                                        if t_stat_p_value_2D < 0.05:
                                            print("p-value (2D; one realization)= ", t_stat_p_value_2D)
                                        #     if Rain_Cloud_show_bool==True:
                                        #         Rain_Cloud_vis(controls_line_proj_dis_2D, patients_line_proj_dis_2D, 'controls', 'patients')
                                    
                                    if D_Emb==3:
                                        t_stat_3D, t_stat_p_value_3D = scipy.stats.ttest_ind(controls_line_proj_dis_3D, patients_line_proj_dis_3D)
                                        p_values_of_N_shuff_3D.append(t_stat_p_value_3D)
                                        if t_stat_p_value_3D < 0.05:
                                            print("p-value (3D; one realization)= ", t_stat_p_value_3D)
                                            if Rain_Cloud_show_bool==True:
                                                Rain_Cloud_vis(controls_line_proj_dis_3D, patients_line_proj_dis_3D, 'controls', 'patients')

            #                         all_t_test_p_values_shuffled1.append(t_stat_p_value)
            #                         if t_stat_p_value > 0.05:
            #                             repulsion_nonsignificant_list.append(gamma)
            #                             alpha_nonsignificant_list.append(alpha)
            #                             n_neighbor_nonsignificant_list.append(n_neighb)
                                
                                if D_Emb==2:
                                    p_values_of_N_shuff_2D= np.array(p_values_of_N_shuff_2D)
                                    #print("Avg p-value (2D)= ", np.mean(p_values_of_N_shuff_2D))
                                    p_values_of_N_boots_Avg_shuff_2D.append(np.mean(p_values_of_N_shuff_2D))
                                if D_Emb==3:
                                    p_values_of_N_shuff_3D= np.array(p_values_of_N_shuff_3D)
                                    #print("Avg p-value (3D)= ", np.mean(p_values_of_N_shuff_3D))
                                    p_values_of_N_boots_Avg_shuff_3D.append(np.mean(p_values_of_N_shuff_3D))

                            if D_Emb==2:
                                p_values_of_N_boots_Avg_shuff_2D= np.array(p_values_of_N_boots_Avg_shuff_2D)
                                #print("p-value for Avg boots Avg shuff= ", np.mean(p_values_of_N_boots_Avg_shuff_2D))
                                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_2D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})        
                                if np.mean(p_values_of_N_boots_Avg_shuff_2D) < 0.05:
                                    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_2D.append(np.mean(p_values_of_N_boots_Avg_shuff_2D))
                                    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_2D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})

                            if D_Emb==3:
                                p_values_of_N_boots_Avg_shuff_3D= np.array(p_values_of_N_boots_Avg_shuff_3D)
                                print("p-value for Avg boots Avg shuff= ", np.mean(p_values_of_N_boots_Avg_shuff_3D))
                                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_3D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})        
                                if np.mean(p_values_of_N_boots_Avg_shuff_3D) < 0.05:
                                    signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_3D.append(np.mean(p_values_of_N_boots_Avg_shuff_3D))
                                    signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D.append({'p_values_of_Avg_boots_shuff': np.mean(p_values_of_N_boots_Avg_shuff_3D),
                                                            'gamma': gamma,
                                                            'alpha': alpha,
                                                            'metricc': metricc,
                                                            'n_neighb': n_neighb,
                                                            'min_dist': min_dist,
                                                            'epoch_n': epoch})
    if D_Emb==2:
        return (signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_2D,
                signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D, #list of dictionaries
                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_2D) #list of dictionaries
    if D_Emb==3:        
        return (signi_t_test_p_val_Avg_boots_Avg_shuff_proj_dis_3D,
                signi_t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D, #list of dictionaries
                t_test_p_val_and_params_Avg_boots_Avg_shuff_proj_dis_3D) #list of dictionaries




#below needs modification
def pval_vs_param_Avg(t_test_p_val_and_params_Avg_shuff_proj_dis,
                        gamma_list,
                        alpha_list,
                        metricc_list,
                        n_neighbor_list,
                        min_dist_list,
                        epochs_list):
    gamma_Avg_of_metric_dic= {}
    alpha_Avg_of_metric_dic= {}
    n_neighbor_Avg_of_metric_dic= {}
    min_dist_Avg_of_metric_dic= {}
    epochs_Avg_of_metric_dic= {}

    #for gamma
    for i,metricc in enumerate(metricc_list):
        gamma_Avg_of_metric_dic[metricc]= []
        for j,gamma in enumerate(gamma_list):
            summ=0
            counter=0
            for k,dic in enumerate(t_test_p_val_and_params_Avg_shuff_proj_dis):
                if dic['metricc']==metricc:
                    if dic['gamma']==gamma:
                        summ += dic['p_values_of_N_seeds']
                        counter += 1
            gamma_Avg_of_metric_dic[metricc].append([gamma,summ/counter])

    #for alpha
    for i,metricc in enumerate(metricc_list):
        alpha_Avg_of_metric_dic[metricc]= []
        for j,alpha in enumerate(alpha_list):
            summ=0
            counter=0
            for k,dic in enumerate(t_test_p_val_and_params_Avg_shuff_proj_dis):
                if dic['metricc']==metricc:
                    if dic['alpha']==alpha:
                        summ += dic['p_values_of_N_seeds']
                        counter += 1
            alpha_Avg_of_metric_dic[metricc].append([alpha,summ/counter])

    #for n_neighbor
    for i,metricc in enumerate(metricc_list):
        n_neighbor_Avg_of_metric_dic[metricc]= []
        for j,n_neighb in enumerate(n_neighbor_list):
            summ=0
            counter=0
            for k,dic in enumerate(t_test_p_val_and_params_Avg_shuff_proj_dis):
                if dic['metricc']==metricc:
                    if dic['n_neighb']==n_neighb:
                        summ += dic['p_values_of_N_seeds']
                        counter += 1
            n_neighbor_Avg_of_metric_dic[metricc].append([n_neighb,summ/counter])

    #for min_dist
    for i,metricc in enumerate(metricc_list):
        min_dist_Avg_of_metric_dic[metricc]= []
        for j,min_dist in enumerate(min_dist_list):
            summ=0
            counter=0
            for k,dic in enumerate(t_test_p_val_and_params_Avg_shuff_proj_dis):
                if dic['metricc']==metricc:
                    if dic['min_dist']==min_dist:
                        summ += dic['p_values_of_N_seeds']
                        counter += 1
            min_dist_Avg_of_metric_dic[metricc].append([min_dist,summ/counter])
                 
    #for epoch
    for i,metricc in enumerate(metricc_list):
        epochs_Avg_of_metric_dic[metricc]= []
        for j,epoch in enumerate(epochs_list):
            summ=0
            counter=0
            for k,dic in enumerate(t_test_p_val_and_params_Avg_shuff_proj_dis):
                if dic['metricc']==metricc:
                    if dic['epoch_n']==epoch:
                        summ += dic['p_values_of_N_seeds']
                        counter += 1
            epochs_Avg_of_metric_dic[metricc].append([epoch,summ/counter])
            
    return(gamma_Avg_of_metric_dic,
            alpha_Avg_of_metric_dic,
            n_neighbor_Avg_of_metric_dic,
            min_dist_Avg_of_metric_dic,
            epochs_Avg_of_metric_dic)


def silhouette(data,   # 2D array with n (number of data points on the plot as sum of two groups) by two
               labels  # 1D array with zeros for first group and ones for second group 
              ):



    # # Assuming you have your data in lists or arrays
    # # For example purposes, I'll generate some random data
    # np.random.seed(42)  # For reproducibility

    # # Generate random data for group A and group B
    # N_A = 15
    # N_B = 25

    # group_A = np.random.rand(N_A, 2)  # 15 points in 2D
    # group_B = np.random.rand(N_B, 2)  # 25 points in 2D

    # # Combine the data into one dataset
    # data = np.vstack((group_A, group_B))

    # # Create labels for the data points
    # labels = np.array([0] * N_A + [1] * N_B)

    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(data, labels)

    # print(f"Silhouette Score: {silhouette_avg:.3f}")

    # Interpretation based on the Silhouette Score
    # if silhouette_avg > 0.5:
    #     interpretation = "The clusters are well-separated."
    # elif silhouette_avg > 0.2:
    #     interpretation = "The clusters are reasonable but could be better."
    # else:
    #     interpretation = "The clusters are not well-defined."

    # print(f"Interpretation: {interpretation}")
    return (silhouette_avg)


def estimate_global_dimension_pca(data,threshold): #threshold like 0.95
    pca = PCA()
    pca.fit(data)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    global_dimension = np.argmax(explained_variance >= threshold) + 1
    return global_dimension


import numpy as np
from sklearn.neighbors import NearestNeighbors

def two_nearest_neighbor_intrinsic_dim(X,N_N): #N_N as number of neighbors
    """
    Estimate intrinsic dimensionality using the Two-Nearest Neighbor (2NN) method.
    
    Parameters:
    - X: np.ndarray, the data matrix (n_samples, n_features)
    
    Returns:
    - float, estimated intrinsic dimensionality
    """
    n, m = X.shape
    nbrs = NearestNeighbors(n_neighbors=N_N).fit(X)  # We need the two nearest neighbors plus the point itself
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Remove the zero distance to self
    
    # Compute the log of the ratio of distances to the second and first nearest neighbors
    log_ratios = np.log(distances[:, 1] / distances[:, 0])
    
    # Estimate intrinsic dimensionality
    intrinsic_dim = 1 / np.mean(log_ratios)
    
    return intrinsic_dim

# # Example usage with a sample dataset
# from sklearn.datasets import load_digits

# # Load a sample dataset
# data = load_digits().data

# # Estimate intrinsic dimensionality
# intrinsic_dim = two_nearest_neighbor_intrinsic_dim(data)
# print(f"Estimated intrinsic dimensionality: {intrinsic_dim}")

def twoNN_from_scikit(data,NN):
    twoNN= skdim.id.TwoNN()
    pw_twoNN=twoNN.fit_pw(data,n_neighbors=NN).dimension_pw_ #point-wise
    #print(pw_twoNN)
    gid_twoNN= twoNN.fit(data).dimension_ #global
    #print(gid_twoNN)
    #plt.plot(pw_twoNN)
    return (gid_twoNN , pw_twoNN)

def CorrInt_from_scikit(data):
    CorrInt= skdim.id.CorrInt()
    gid_CorrInt= CorrInt.fit(data).dimension_ #global
    return gid_CorrInt

def DANCo_from_scikit(data):
    DANCo= skdim.id.DANCo()
    gid_DANCo= DANCo.fit(data).dimension_ #global
    return gid_DANCo

def ESS_from_scikit(data):
    ESS= skdim.id.ESS()
    gid_ESS= ESS.fit(data).dimension_ #global
    return gid_ESS

def KNN_from_scikit(data):
    KNN= skdim.id.KNN()
    gid_KNN= KNN.fit(data).dimension_ #global
    return gid_KNN

def MADA_from_scikit(data):
    MADA= skdim.id.MADA()
    gid_MADA= MADA.fit(data).dimension_ #global
    return gid_MADA

def MiND_MLk_from_scikit(data):
    MiND= skdim.id.MiND_ML(ver='MLk')
    gid_MiND= MiND.fit(data).dimension_ #global
    return gid_MiND

def MiND_MLk_from_scikit1(data, kk=20, DD=10):
    MiND= skdim.id.MiND_ML(k=kk, D=DD, ver='MLk') 
    gid_MiND= MiND.fit(data).dimension_ #global
    return gid_MiND

def MiND_MLi_from_scikit(data):
    MiND= skdim.id.MiND_ML(ver='MLi')
    gid_MiND= MiND.fit(data).dimension_ #global
    return gid_MiND

def MiND_MLi_from_scikit1(data, kk=20, DD=10):
    MiND= skdim.id.MiND_ML(k=kk, D=DD, ver='MLi') 
    gid_MiND= MiND.fit(data).dimension_ #global
    return gid_MiND

def MOM_from_scikit(data):
    MOM= skdim.id.MOM()
    gid_MOM= MOM.fit(data).dimension_ #global
    return gid_MOM

def TLE_from_scikit(data):
    TLE= skdim.id.TLE()
    gid_TLE= TLE.fit(data).dimension_ #global
    return gid_TLE



def MLE_from_scikit(data):
    MLE= skdim.id.MLE()
    #pw_twoNN=twoNN.fit_pw(data,n_neighbors=NN).dimension_pw_ #point-wise
    #print(pw_twoNN)
    gid_MLE= MLE.fit(data).dimension_ #global
    #print(gid_twoNN)
    #plt.plot(pw_twoNN)
    return gid_MLE


def  FisherSep_from_scikit(data,NN):
    FisherSep= skdim.id.FisherS()
    pw_FisherSep= FisherSep.fit_pw(data, n_neighbors=NN).dimension_pw_ #point-wise
    #print(pw_twoNN)
    gid_FisherSep= FisherSep.fit(data).dimension_ #global
    #print(gid_twoNN)
    #plt.plot(pw_twoNN)
    return (gid_FisherSep , pw_FisherSep)

def lpca_from_scikit(data):
    lpca= skdim.id.lPCA()
    gid_lpca= lpca.fit(data).dimension_ #global
    return gid_lpca





def mannwhitneyu_stat(group1,group2):
# Mann-Whitney U test
    stat, p = mannwhitneyu(group1, group2)
    #print('Mann-Whitney U Test: Statistic=%.3f, p=%.3f' % (stat, p))

    # if p > 0.05:
    #     print('Probably the same distribution (fail to reject H0)')
    # else:
    #     print('Probably different distributions (reject H0)')
    return p

def Benjamini_Hochberg_FDR_control(p_vals_list, FDR_control_level=0.05, plot_bool= False):

    # print("p_values:")
    # print(p_vals_list)

    p_vals_list.sort()

    # Plot the sorted values versus their indices
    if plot_bool==True:
        plt.plot(range(len(p_vals_list)), p_vals_list, marker='o')
        plt.show()

    p_vals_pairs= []
    for k,pval in enumerate(p_vals_list):
        p_vals_pairs.append([pval,(k+1) * FDR_control_level * (1/len(p_vals_list))])

    # print("p_value pairs:")
    # print(p_vals_pairs)

    signi_p_vals= []
    for pair in p_vals_pairs:
        if pair[0] <= pair[1]:
            signi_p_vals.append(pair[0])


    # print("significant p_values due to Benjamini-Hochberg procedure :")
    # print(signi_p_vals)
    return signi_p_vals


