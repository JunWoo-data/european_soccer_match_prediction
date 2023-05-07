# %%
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

# %%
def my_histogram(df, column_name):
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {"height_ratios": (.2, .8)}, figsize = (10, 7))

    sns.boxplot(x = df[column_name], ax = ax_box, showfliers = False)
    sns.histplot(x = df[column_name], ax = ax_hist, kde = True)

    plt.grid()
    plt.xlabel(column_name, fontsize = 16)
    plt.ylabel("Count", fontsize = 16)
    ax_box.set_xlabel("")

    plt.show()
    
# %%
def pca_results(data, pca):
    
    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys()) 
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1) 
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance']) 
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar', width = 0.75)
    ax.set_ylabel("Feature Weights", fontsize = 15) 
    ax.set_xticklabels(dimensions, rotation = 0, fontsize = 15)
    ax.legend(bbox_to_anchor = (1.2, 1), loc = 'upper right')

    # Display the explained variance ratios# 
    for i, ev in enumerate(pca.explained_variance_ratio_): 
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev), fontsize = 15)
    
    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)
# %%
