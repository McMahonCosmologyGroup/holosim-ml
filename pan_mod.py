"""
Panel definitions of the SO LAT.

Grace E. Chesmore
May 2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_per_m1 = [7, 9, 9, 9, 9, 9, 9, 9, 7]  # Num panels per row
n_per_m2 = [5, 7, 9, 9, 9, 9, 9, 7, 5]  # Num panels per row

# Initializes the pattern of panels in
# the primary and secondary mirrors.
def initialize_panel_model():

    out_m1 = np.ones((10, sum(n_per_m1)))
    out_m2 = np.ones((10, sum(n_per_m2)))

    i_out = 0
    i_row = 0
    while i_row < 9:
        i = 0
        while i < n_per_m1[i_row]:
            out_m1[0, i_out] = i_row + 1  # row
            out_m1[1, i_out] = i + 1  # column
            i_out += 1
            i += 1
        i_row += 1

    i_out = 0
    i_row = 0
    while i_row < 9:
        i = 0
        while i < n_per_m2[i_row]:
            out_m2[0, i_out] = i_row + 1  # row
            out_m2[1, i_out] = i + 1  # column
            i_out += 1
            i += 1
        i_row += 1

    # The central row of adjusters in the adjuster positions .csv
    # file are out of order (for both the primary and secondary),
    # so here I re-order them to match the rest of the panels in the mirrors.
    out_m1[1, 0:7] += 1
    out_m1[1, 70:77] += 1
    out_m2[1, 0:5] += 2
    out_m2[1, 5:12] += 1
    out_m2[1, 57:64] += 1
    out_m2[1, 64:69] += 2

    return out_m1, out_m2


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Given an x and y position, the function determines
# which panel that point sits on.
def identify_panel(x, y, x_adj, y_adj, col, row):

    n_pts = len(x)
    out = np.ones((2, n_pts, n_pts))
    for ii in range(n_pts):
        for jj in range(n_pts):
            indx_x = find_nearest(x_adj, x[ii, jj])
            indx_y = find_nearest(y_adj, y[ii, jj])

            if x[ii, jj] < np.min(x_adj):
                out[1, ii, jj] = 0

            elif x[ii, jj] > np.max(x_adj):
                out[1, ii, jj] = 10

            elif y[ii, jj] < np.min(y_adj):
                out[0, ii, jj] = 0

            elif y[ii, jj] > np.max(y_adj):
                out[0, ii, jj] = 10
            else:
                out[0, ii, jj] = row[indx_x]
                out[1, ii, jj] = col[indx_y]
    return out


# Defines the panel model given specified
# adjuster offsets.
def panel_model_from_adjuster_offsets(mirror, err, num, save):

    if num == 0:
        delta_z = np.random.randn(1092) * err / 1e3
    else:
        delta_z = err / 1e3

    # If you want to save the specific adjuster offsets
    # define the path of this saved file here:
    if save != 0:
        np.savetxt(
            "/data/chesmore/sim_out/rx00600/adj_offsets_m"
            + str(mirror)
            + "_"
            + str(num)
            + ".txt",
            np.c_[delta_z],
        )

    pan_mod1, pan_mod2 = initialize_panel_model()
    adj_pos_m1, adj_pos_m2 = get_single_vert_adj_positions()

    if mirror == 1:
        npan = sum(n_per_m1)
        pan_mod = pan_mod1
        col = adj_pos_m1[0]
        row = adj_pos_m1[1]
        x_adj = adj_pos_m1[2]
        y_adj = adj_pos_m1[3]
        z_adj = adj_pos_m1[4]
    else:
        npan = sum(n_per_m2)
        pan_mod = pan_mod2
        col = adj_pos_m2[0]
        row = adj_pos_m2[1]
        x_adj = adj_pos_m2[4]
        y_adj = adj_pos_m2[3]
        z_adj = adj_pos_m2[2]

    i = 0
    while i < npan:
        i_row = pan_mod[0, i]  # Row number
        i_col = pan_mod[1, i]  # Column number

        # The calculation of the model parameters
        cur_adj = np.where((row == i_row) & (col == i_col))

        if len(cur_adj[0]) == 0:
            i += 1
            continue
        else:

            n_adj = len(cur_adj[0])
            dz_cur = delta_z[cur_adj]  # dz error

            x_cur = x_adj[cur_adj]
            y_cur = y_adj[cur_adj]

            mean_x = np.mean(x_cur)
            mean_y = np.mean(y_cur)

            x_cur = x_cur - mean_x
            y_cur = y_cur - mean_y

            if i_col == 5:
                x_cur = np.array([x_cur[0], x_cur[3], x_cur[1], x_cur[4], x_cur[2]])
                y_cur = np.array([y_cur[0], y_cur[3], y_cur[1], y_cur[4], y_cur[2]])

            M = np.zeros((5, 5))
            M[:, 0] = 1.0
            M[:, 1] = x_cur
            M[:, 2] = y_cur
            M[:, 3] = x_cur ** 2 + y_cur ** 2
            M[:, 4] = y_cur * x_cur

            coeffs = np.matmul(np.linalg.inv(M), dz_cur)

            a = coeffs[0]
            b = coeffs[1]
            c = coeffs[2]
            d = coeffs[3]
            e = coeffs[4]
            f = 0

            # The output of the model
            pan_mod[2, i] = a
            pan_mod[3, i] = b
            pan_mod[4, i] = c
            pan_mod[5, i] = d
            pan_mod[6, i] = e
            pan_mod[7, i] = f
            pan_mod[8, i] = mean_x
            pan_mod[9, i] = mean_y

        i += 1

    return pan_mod


# Constructs the z offset given adjuster offsets.
def reconstruct_z_from_pan_model(x, y, x_adj, y_adj, col, row, panm, mirror):

    de_z = np.zeros(np.shape(x))
    print("Starting panel identification...")
    pan_id = identify_panel(x, y, x_adj, y_adj, col, row)  # returns row and column #'s
    print("Panel identification finished.")
    print("Starting panel reconstruction...")

    i = 0
    while i < len(x):
        j = 0
        while j < len(x):

            cur_pan = np.where(
                (panm[0, :] == pan_id[0, i, j]) & (panm[1, :] == pan_id[1, i, j])
            )
            if len(cur_pan[0]) == 0:
                de_z[i, j] = np.nan
            else:

                if mirror == 2:
                    x_edge = x_edge_m2[int(cur_pan[0])]
                    y_edge = y_edge_m2[int(cur_pan[0])]
                else:
                    x_edge = x_edge_m1[int(cur_pan[0])]
                    y_edge = y_edge_m1[int(cur_pan[0])]

                if (
                    x[i, j] > (np.max(x_adj) + x_edge)
                    or x[i, j] < (np.min(x_adj) - x_edge)
                    or y[i, j] > (np.max(y_adj) + y_edge)
                    or y[i, j] < (np.min(y_adj) - y_edge)
                ):
                    de_z[i, j] = np.nan

                else:

                    xc = panm[8, cur_pan]
                    yc = panm[9, cur_pan]

                    xi = x[i, j] - xc
                    yi = y[i, j] - yc

                    a = panm[2, cur_pan]
                    b = panm[3, cur_pan]
                    c = panm[4, cur_pan]
                    d = panm[5, cur_pan]
                    e = panm[6, cur_pan]

                    de_z[i, j] = (
                        a
                        + (b * xi)
                        + (c * yi)
                        + d * (xi ** 2 + yi ** 2)
                        + (e * xi * yi)
                    )
            j += 1
        i += 1

    print("Finished panel reconstruction.")
    return de_z


# Vertex adjuster positions are read in
# from .csv file.
def get_single_vert_adj_positions():

    out_m1 = []
    out_m2 = []

    # Primary mirror adjuster positions
    df_m1 = pd.read_csv(
        "/home/chesmore/Desktop/Code/holosim_paper/package/holosim-ml/pans-adjs/Mirror-M1-vertical-adjuster-points_r1-1.csv",
        skiprows=2,
        na_values=["<-- ", "--> ", "<--", "-->"],
    )
    # Secondary mirror adjuster positions
    df_m2 = pd.read_csv(
        "/home/chesmore/Desktop/Code/holosim_paper/package/holosim-ml/pans-adjs/Mirror-M2-vertical-adjuster-points_r1-1.csv",
        skiprows=2,
        na_values=["<-- ", "--> ", "<--", "-->"],
    )

    # Read in adjuster positions from columns
    x_adj_m1 = df_m1["X2"]
    y_adj_m1 = df_m1["Y2"]
    z_adj_m1 = df_m1["Z2"]
    x_adj_m1_2 = df_m1["x "]
    y_adj_m1_2 = df_m1["y "]
    z_adj_m1_2 = df_m1["z "]
    x_adj_m2 = df_m2["X2"]
    y_adj_m2 = df_m2["Y2"]
    z_adj_m2 = df_m2["Z2"]
    x_adj_m2_2 = df_m2["x"]
    y_adj_m2_2 = df_m2["y"]
    z_adj_m2_2 = df_m2["z"]
    xx = x_adj_m1[x_adj_m1.notnull()]
    yy = y_adj_m1[x_adj_m1.notnull()]
    zz = z_adj_m1[x_adj_m1.notnull()]
    xx_2 = x_adj_m1_2[x_adj_m1_2.notnull()]
    yy_2 = y_adj_m1_2[x_adj_m1_2.notnull()]
    zz_2 = z_adj_m1_2[x_adj_m1_2.notnull()]
    ##################################
    pan_id = df_m1["Panel2"]
    pan_id_2 = df_m1["Panel "]
    pp = pan_id[x_adj_m1.notnull()]
    pp_2 = pan_id_2[x_adj_m1_2.notnull()]
    pp = ["%.0f" % number for number in pp]
    pan_id_m1 = np.concatenate((pp_2, pp))
    ##################################
    x_adj_m1 = np.array(xx)
    y_adj_m1 = np.array(yy)
    z_adj_m1 = np.array(zz)
    x_adj_m1_2 = np.array(xx_2)
    y_adj_m1_2 = np.array(yy_2)
    z_adj_m1_2 = np.array(zz_2)

    x_adj_m1 = np.concatenate((x_adj_m1_2, x_adj_m1))
    y_adj_m1 = np.concatenate((y_adj_m1_2, y_adj_m1))
    z_adj_m1 = np.concatenate((z_adj_m1_2, z_adj_m1))

    xx = x_adj_m2[x_adj_m2.notnull()]
    yy = y_adj_m2[x_adj_m2.notnull()]
    zz = z_adj_m2[x_adj_m2.notnull()]
    xx_2 = x_adj_m2_2[x_adj_m2_2.notnull()]
    yy_2 = y_adj_m2_2[x_adj_m2_2.notnull()]
    zz_2 = z_adj_m2_2[x_adj_m2_2.notnull()]
    ##################################
    pan_id = df_m2["Panel2"]
    pan_id_2 = df_m2["Panel"]
    pp = pan_id[x_adj_m2.notnull()]
    pp_2 = pan_id_2[x_adj_m2_2.notnull()]
    pan_id_m2 = np.concatenate((pp_2, pp))
    pan_id_m2 = ["%.0f" % number for number in pan_id_m2]
    ##################################
    x_adj_m2 = np.array(xx)
    y_adj_m2 = np.array(yy)
    z_adj_m2 = np.array(zz)
    x_adj_m2_2 = np.array(xx_2)
    y_adj_m2_2 = np.array(yy_2)
    z_adj_m2_2 = np.array(zz_2)

    x_adj_m2 = np.concatenate((x_adj_m2_2, x_adj_m2))
    y_adj_m2 = np.concatenate((y_adj_m2_2, y_adj_m2))
    z_adj_m2 = np.concatenate((z_adj_m2_2, z_adj_m2))

    # Primary column and row numbers
    col_m1 = np.zeros(len(pan_id_m1))
    row_m1 = np.zeros(len(pan_id_m1))
    for kk in range(len(pan_id_m1)):
        pan = pan_id_m1[kk]
        col_m1[kk] = pan[3]
        row_m1[kk] = pan[2]

    # Secondary column and row numbers
    col_m2 = np.zeros(len(pan_id_m2))
    row_m2 = np.zeros(len(pan_id_m2))

    for kk in range(len(pan_id_m2)):
        pan = pan_id_m2[kk]
        col_m2[kk] = pan[3]
        row_m2[kk] = pan[2]

    out_m1.append(col_m1)
    out_m1.append(row_m1)
    out_m1.append(x_adj_m1)
    out_m1.append(y_adj_m1)
    out_m1.append(z_adj_m1)
    out_m2.append(col_m2)
    out_m2.append(row_m2)
    out_m2.append(x_adj_m2)
    out_m2.append(y_adj_m2)
    out_m2.append(z_adj_m2)
    return out_m1, out_m2
