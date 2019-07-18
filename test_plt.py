import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
from PyQt5.QtWidgets import QApplication, QFileDialog

import yaml

def bin(val, binned_lst):
    assert val <= max(binned_lst[-1]), "Invalid input. Input value exceeds maximum bin value."
    low, high = 0, len(binned_lst)-1
    binned = False
    while not binned:
        mid = (high+low)//2
        if high <= low or in_bin(val, binned_lst[mid]):
            binned = True
        elif val < binned_lst[mid][0]:
            high = mid - 1
        else:
            low = mid + 1
    return mid

def in_bin(val, bin):
    if val >= bin[0] and val < bin[1]:
        return True
    else:
        return False

def create_bins(sorted_lst):
    return [(sorted_lst[i], sorted_lst[i+1]) for i in range(len(sorted_lst)-1)]

def heatmap(data, x_title, y_title, row_labels, col_labels, ax=None, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.set_title("Number of Aggregations",y=-2,pad=-1, size='small')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    x_range = np.arange(data.shape[1]+1).astype(np.float)
    x_range += np.full(x_range.shape,-0.5)

    y_range = np.arange(data.shape[0]+1).astype(np.float)
    y_range += np.full(y_range.shape,-0.5)

    ax.set_xticks(x_range)
    ax.set_yticks(y_range)
    ax.set_xticklabels(col_labels, size='smaller')
    ax.set_yticklabels(row_labels, size='smaller')

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, pad=-2, color='white')

    ax.tick_params(axis='x', labelrotation=30)
    ax.tick_params(axis='y', labelrotation=30)

    ax.set_xlabel(x_title, size='small')
    ax.xaxis.set_label_position('top')

    ax.set_ylabel(y_title,size='small')


    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def plot_heatmap(data_path, id_col_name, x_col_name, y_col_name, x_label_range, y_label_range,
                    x_ticks=10, y_ticks=10, x_exp=True, y_exp=False, x_base=2, y_base=2, x_max_data_val=None, y_max_data_val=None,
                    x_min_data_val=None, y_min_data_val=None, colorbar_label='', color_map='coolwarm'):

    data_file = data_path

    try:
        with open(data_file) as f:
             df = pd.read_csv(f, usecols=[id_col_name, x_col_name, y_col_name])
    except:
        with open(data_file, encoding='utf16') as f:
             df = pd.read_csv(f, usecols=[id_col_name, x_col_name, y_col_name])

    x_values = df[x_col_name].unique()
    y_values = df[y_col_name].unique()

    x_label_min, x_label_max = x_label_range[0], x_label_range[1]
    y_label_min, y_label_max = y_label_range[0], y_label_range[1]

    if x_exp:
        if x_label_min == 0:
            x_label_min = x_base
        x_label_min = math.ceil(math.log(x_label_min, x_base))
        x_label_max = math.ceil(math.log(x_label_max, x_base))
        x_labels = [0]+[x_base**i for i in range(x_label_min, x_label_max+1)]
    else:
        x_labels = np.linspace(x_label_min, x_label_max, x_ticks)
        x_labels = [round(num,2) for num in x_labels]
        assert all(i < j for i, j in zip(x_labels, x_labels[1:])), \
                "Error: X label list must be strictly increasing when rounded to 2nd decimal. Try decreasing number of x ticks"


    if y_exp:
        if y_label_min == 0:
            y_label_min = y_base
        y_label_min = math.ceil(math.log(y_label_min, y_base))
        y_label_max = math.ceil(math.log(y_label_max, y_base))
        y_labels = [0]+[y_base**i for i in range(y_label_min, y_label_max+1)]
    else:
        y_labels = np.linspace(y_label_min,y_label_max,y_ticks)
        y_labels = [round(num,2) for num in y_labels]

        assert all(i < j for i, j in zip(y_labels, y_labels[1:])), \
                "Error: Y label list must be strictly increasing when rounded to 2nd decimal. Try decreasing number of y ticks"


    x_range = create_bins(x_labels)
    y_range = create_bins(y_labels)
    if x_max_data_val == None:
        x_max_data_val = max(x_labels)
    if y_max_data_val == None:
        y_max_data_val = max(y_labels)
    if x_min_data_val == None:
        x_min_data_val = min(x_labels)
    if y_min_data_val == None:
        y_min_data_val = min(y_labels)

    filtered_df = df.loc[(df[x_col_name] <= x_max_data_val) & (df[y_col_name] <= y_max_data_val)].groupby([x_col_name, y_col_name])[id_col_name].nunique()

    mapping = np.zeros((len(x_range), len(y_range)))

    for key in dict(filtered_df):
        mapping[bin(key[0], x_range), bin(key[1], y_range)] += filtered_df[key]

    im, _ = heatmap(np.transpose(mapping), x_col_name, y_col_name, y_labels, x_labels, cmap=color_map, cbarlabel=colorbar_label)

    plt.show()

if __name__ == "__main__":
    """
    #path to csv file
    App = QApplication(sys.argv)
    window = Window(title="Heatmap Settings", top=300, left=800, width=400, height=300)
    sys.exit(App.exec())
    """
    app = QApplication([])
    config_filename = QFileDialog.getOpenFileName(None, caption="Select configuration file location.")[0]

    with open(config_filename, 'r') as config_file:
        cfg = yaml.safe_load(config_file)
    
    data_file = cfg['csv_file']

    """ Information Specifying Column Names """ # must match names in csv file
    id_col_name = cfg['id_column_name']
    x_col_name = cfg['x_axis_info']['column_name']
    y_col_name = cfg['y_axis_info']['column_name']

    """ Heatmap Information """
    x_label_range = [cfg['x_axis_info']['labels_min'], cfg['x_axis_info']['labels_max']] # range of values displayed on heatmap
    y_label_range = [cfg['y_axis_info']['labels_min'], cfg['y_axis_info']['labels_max']]

    x_ticks = cfg['x_axis_info']['num_ticks'] # number of bins in between range of values (graph gets weird if you make them too big). Does not apply for log scales.
    y_ticks = cfg['y_axis_info']['num_ticks']

    x_max_data_val = cfg['x_axis_info']['data_max'] # Number that thresholds data (for example, if you only wanted up until 128). Must be less than the respective label range
    y_max_data_val = cfg['y_axis_info']['data_max']

    x_min_data_val = cfg['x_axis_info']['data_min']
    y_min_data_val = cfg['y_axis_info']['data_min']

    colorbar_label = cfg['colorbar_label']
    cmap = cfg['color_map']

    x_log_scale = cfg['x_axis_info']['logarithmic'] # use logarithmic scale
    y_log_scale = cfg['y_axis_info']['logarithmic']

    x_base = cfg['x_axis_info']['base'] # use base 2 for log x_log_scale
    y_base = cfg['y_axis_info']['base']

    #make_gui()
    plot_heatmap(data_file, id_col_name, x_col_name, y_col_name, x_label_range, y_label_range, x_ticks=x_ticks,
                    y_ticks=y_ticks, x_exp=x_log_scale, y_exp=y_log_scale, x_base=x_base, x_max_data_val=x_max_data_val, y_max_data_val=y_max_data_val,
                    x_min_data_val=x_min_data_val, y_min_data_val=y_min_data_val, colorbar_label=colorbar_label, color_map=cmap)
