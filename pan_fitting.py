"""
Extra panel definitions of the SO LAT.

Grace E. Chesmore
May 2021
"""

import numpy as np
import pan_mod as pm
from pan_mod import *
from tele_geo import *


def find_nearest_spot(array1, array2, value1, value2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    idx = (np.abs(array2 - value2) + np.abs(array1 - value1)).argmin()
    return idx


def pannel_mask(x, y, err, out, plotting):

    err_new = err
    adj_pos1, adj_pos2 = pm.get_single_vert_adj_positions()

    y_mask = (out[4, :] - np.mean(out[4, :])) / 1e3
    x_mask = (out[3, :]) / 1e3
    mask = np.ones(np.shape(y_mask))

    for ii in range(len(x)):
        for jj in range(len(y)):
            val_x = x[ii, jj]
            val_y = y[ii, jj]

            arr_x = x_mask
            arr_y = y_mask

            indx = find_nearest_spot(arr_x, arr_y, val_x, val_y)

            resid = np.sqrt((arr_x[indx] - val_x) ** 2 + (arr_y[indx] - val_y) ** 2)
            if resid > 0.05:
                err_new[ii, jj] = np.nan

    return x, y, err_new


###### Panel Edges ###########
n_per_m1 = [7, 9, 9, 9, 9, 9, 9, 9, 7]  # num panels per row
n_per_m2 = [5, 7, 9, 9, 9, 9, 9, 7, 5]  # num panels per row

npan = np.sum(n_per_m1)
x_edge_m1 = np.ones(npan)
y_edge_m1 = np.ones(npan)

pan_mod1, pan_mod2 = pm.initialize_panel_model()
adj_pos_m1, adj_pos_m2 = pm.get_single_vert_adj_positions()
col = adj_pos_m1[0]
row = adj_pos_m1[1]
x_adj = adj_pos_m1[2]
y_adj = adj_pos_m1[3]
z_adj = adj_pos_m1[4]

npan = np.sum(n_per_m1)
i = 0
while i < npan:
    i_row = pan_mod1[0, i]  # Row number
    i_col = pan_mod1[1, i]  # Column number

    # The calculation of the model parameters
    cur_adj = np.where((row == i_row) & (col == i_col))
    if len(cur_adj[0]) == 0:
        i += 1
        continue
    else:

        x_cur = x_adj[cur_adj]
        y_cur = y_adj[cur_adj]

        x_edge_m1[i] = (710 - (np.max(x_cur) - np.min(x_cur))) / 2
        y_edge_m1[i] = (710 - (np.max(y_cur) - np.min(y_cur))) / 2

    i += 1

col = adj_pos_m2[0]
row = adj_pos_m2[1]
x_adj = adj_pos_m2[4]
y_adj = adj_pos_m2[3]
z_adj = adj_pos_m2[2]

npan = np.sum(n_per_m2)
x_edge_m2 = np.ones(npan)
y_edge_m2 = np.ones(npan)

i = 0
while i < npan:
    i_row = pan_mod2[0, i]  # Row number
    i_col = pan_mod2[1, i]  # Column number

    # The calculation of the model parameters
    cur_adj = np.where((row == i_row) & (col == i_col))
    if len(cur_adj[0]) == 0:
        i += 1
        continue
    else:

        x_cur = x_adj[cur_adj]
        y_cur = y_adj[cur_adj]

        x_edge_m2[i] = (710 - (np.max(x_cur) - np.min(x_cur))) / 2
        y_edge_m2[i] = (710 - (np.max(y_cur) - np.min(y_cur))) / 2
    i += 1
