import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def FC_matrix_dfs_plot(df_FC_list,
                       titles_list,
                       x_label,
                       y_label,
                       PNGsave_bool,
                       save_path_and_name=None):
    for j, df_FC in enumerate(df_FC_list):
        # Plotting the correlation matrix as a heatmap
        plt.figure(figsize=(4, 3.2))
        sns.heatmap(df_FC, annot=False, cbar=True, fmt=".2f", cmap='jet', center=0)
        # plt.title(f'FC (Pearson Correlation) of portion #{j} of simulated Time Series')
        plt.title(titles_list[j])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if PNGsave_bool==True:
            plt.savefig(save_path_and_name.format(j), dpi=300)
        plt.show()


def visualize_components(component1, component2, labels,i,j, text, show=True):

    mpl.rcParams['font.family'] = 'serif'
    plt.figure()
    #plt.plot(component1, component2, '-o')
    plt.scatter(x=component1, y=component2, c=labels, cmap='cool', s=3)
    #plt.scatter(x=component1, y=component2, c=labels, cmap='tab10', s=1)

    plt.xlabel(fr"$\psi_{{{i}}}$")
    plt.ylabel(fr"$\psi_{{{j}}}$")

    plt.text(0.95, 0.95, text, transform=plt.gca().transAxes,
        fontsize=8, verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='yellow', alpha=0.4)) #fontsize=10; alpha=0.8
    plt.colorbar(ticks=[0,1])
    #plt.colorbar(ticks=[0,10,20,30,40,50,60,70,80])
    #plt.clim(-0.5, 9.5)
    if show:
        plt.show()
    

#plt.rcParams['figure.dpi'] = 150
#mpl.rcParams['font.family'] = 'serif'


# Set random seed for reproducibility
#np.random.seed(0)
def visualize_6FC_components(component1, component2, labels,i,j, text, show=True):
    ii=i
    jj=j
    
    controls_component1= []
    controls_component2= []
    controls_labels= []
    
    patients_component1= []
    patients_component2= []
    patients_labels= []
    
    for i,code in enumerate(labels):
        if code[0] == 'c':
            controls_component1.append(component1[i])
            controls_component2.append(component2[i])
            controls_labels.append(labels[i])       
    
        elif code[0] == 'p':
            patients_component1.append(component1[i])
            patients_component2.append(component2[i])
            patients_labels.append(labels[i])    

# participant_id = int(code[1:3])  
# image_number = int(code[3])   

    # Create the colormap
    cmap = plt.get_cmap('nipy_spectral')
    # Normalize the numeric_labels to the range of the colormap
    norm = plt.Normalize(1, 17)
    # Plotting
    fig, ax = plt.subplots()
    for (i, label) in enumerate(controls_labels):
        ax.scatter(controls_component1[i], controls_component2[i],
                   color=cmap(norm(int(label[1:3]))),
                   marker=f'${int(label[3])}$'+'$.$', s=20)
        
    norm = plt.Normalize(1, 34)
    for (i, label) in enumerate(patients_labels):
        ax.scatter(patients_component1[i], patients_component2[i],
                   color=cmap(norm(int(label[1:3]))),
                   marker=f'${label[3]}$', s=20)
        
    # Add a colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
    # cbar = plt.colorbar(sm, ticks=np.linspace(1, 34, 4))
    # cbar.set_label('Label Value')
    # Set plot details
    # ax.set_title('Scatter Plot with Text Markers and Color Mapping')
    
    plt.xlabel(fr"$\psi_{{{ii}}}$")
    plt.ylabel(fr"$\psi_{{{jj}}}$")
    
    #plt.savefig('/Users/user/NIBS 2 patients Schaeffer300/high_res_plot.png', dpi=500)
    plt.show()

def Rain_Cloud_vis(data1,
                   data2,
                   label1,
                   label2,
                   x_label,
                   y_label,
                   title):

    # Combine into a single data structure and create a corresponding labels array
    data = np.concatenate([data1, data2])
    labels = [label1] * len(data1) + [label2] * len(data2)

    # Set style
    sns.set(style="whitegrid")

    # Initialize the figure
    plt.figure(figsize=(8, 5))

    # Create a violin plot
    ax = sns.violinplot(x=labels, y=data, inner=None, color="0.8")

    # Add a strip plot with jitter
    sns.stripplot(x=labels, y=data, jitter=True, size=3, color='red', edgecolor="pink") #color='black', edgecolor="gray")

    # Calculate the means and quartiles, then plot them
    means = [np.mean(data1), np.mean(data2)]
    q1 = [np.percentile(data1, 25), np.percentile(data2, 25)]
    q3 = [np.percentile(data1, 75), np.percentile(data2, 75)]

    # Use bar plot for means and error bars for quartiles
    plt.bar([label1, label2], means, color='red', alpha=0.0, yerr=[np.abs(np.array(q1)-means), np.abs(np.array(q3)-means)], capsize=5)

    # Enhance plot details
    plt.title(title) #'Rain Cloud Plot with Mean and Quartile Bars'
    plt.xlabel(x_label) #'Group'
    plt.ylabel(y_label) #'Value Distribution'

    # Show the plot
    plt.show()



def Rain_Cloud_vis1(data1,
                   data2,
                   label1,
                   label2,
                   x_label,
                   y_label,
                   title,
                   backgroung_color,
                   save_path=None):

    # Combine into a single data structure and create a corresponding labels array
    data = np.concatenate([data1, data2])
    labels = [label1] * len(data1) + [label2] * len(data2)

    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 5), dpi=400)
    plt.gca().set_facecolor((backgroung_color, 0.4))  # Set the background color of the plot #'lightgreen'

    # Create a violin plot with custom colors for each group
    ax = sns.violinplot(x=labels, y=data, inner=None, palette={"controls": "lightcoral", "patients": "deepskyblue"})

    # Add a strip plot with jitter and custom colors
    sns.stripplot(x=labels, y=data, jitter=True, size=3, palette={"controls": "red", "patients": "blue"}, edgecolor="pink")

    # Calculate the means and quartiles, then plot them
    means = [np.mean(data1), np.mean(data2)]
    q1 = [np.percentile(data1, 25), np.percentile(data2, 25)]
    q3 = [np.percentile(data1, 75), np.percentile(data2, 75)]

    # Use bar plot for means and error bars for quartiles
    plt.bar([label1, label2], means, color='black', alpha=0.0, yerr=[np.abs(np.array(q1)-means), np.abs(np.array(q3)-means)], capsize=5)

    # Add lines for quartiles and mean
    for i, (quart1, mean, quart3) in enumerate(zip(q1, means, q3)):
        plt.plot([i-0.05, i+0.05], [quart1, quart1], color='black', linewidth=1)
        plt.plot([i-0.1,  i+0.1],  [mean, mean], color='black', linewidth=3)
        plt.plot([i-0.05, i+0.05], [quart3, quart3], color='black', linewidth=1)


    #plt.ylim(2, 7)

    # Enhance plot details
    plt.title(title) #'Rain Cloud Plot with Mean and Quartile Bars'
    plt.xlabel(x_label) #'Group'
    plt.ylabel(y_label) #'Value Distribution'

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')

    # Show the plot
    plt.show()

