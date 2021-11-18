import sys

import numpy as np

sys.path.append("/home/chesmore/Desktop/Code/holosim_paper/package/holosim-ml")

import pickle

import ap_field as af
import ap_fitting as afit
import far_field as ff
import optics_analyze as oa
import pan_mod as pm
import sklearn
import tele_geo as tg
from sklearn.linear_model import LinearRegression

save = 0

# Trinocular positions
rx_x = np.array([0, 0, 0])
rx_z = np.array([0, 0, 0])
el = np.array([oa.el_offset(rx_z[0]), oa.el_offset(rx_z[1]), oa.el_offset(rx_z[2])])
az = np.array([oa.az_offset(rx_x[0]), oa.az_offset(rx_x[1]), oa.az_offset(rx_x[2])])

shift_C = ["y", oa.sh_z(rx_z[2])]

# Trinocular positions
rx_x_tri = np.array([-519.62 * (3 / 2), 519.62 * (3 / 2), 0])
rx_z_tri = np.array([-300 * (3 / 2), -300 * (3 / 2), 600 * (3 / 2)])
el_tri = np.array(
    [oa.el_offset(rx_z_tri[0]), oa.el_offset(rx_z_tri[1]), oa.el_offset(rx_z_tri[2])]
)
az_tri = np.array(
    [oa.az_offset(rx_x_tri[0]), oa.az_offset(rx_x_tri[1]), oa.az_offset(rx_x_tri[2])]
)

shift_A_tri = ["xy", oa.sh_x(rx_x_tri[0]), oa.sh_z(rx_z_tri[0])]
shift_B_tri = ["xy", oa.sh_x(rx_x_tri[1]), oa.sh_z(rx_z_tri[1])]
shift_C_tri = ["y", oa.sh_z(rx_z_tri[2])]

n_adj_m1 = 5 * 77
n_adj_m2 = 5 * 69


def tele_geo_init(x, y, z, el, az):
    tele_geo = tg.initialize_telescope_geometry()
    tele_geo.rx_x = x
    tele_geo.rx_y = y
    tele_geo.rx_z = z
    tele_geo.el0 += el
    tele_geo.az0 += az
    return tele_geo


rx3 = np.array([rx_x[2], 209.09, rx_z[2]])
tele_geo = tele_geo_init(rx3[0], rx3[1], rx3[2], el[2], az[2])

th = np.linspace(-np.pi / 2 - 0.28, -np.pi / 2 + 0.28, tele_geo.N_scan)
ph = np.linspace(np.pi / 2 - 0.28, np.pi / 2 + 0.28, tele_geo.N_scan)

rxmirror_C = af.ray_mirror_pts(rx3, tele_geo, th, ph)
dat_C = afit.take_measurement(
    np.zeros(77 * 5), np.zeros(77 * 5), 0, tele_geo, rxmirror_C
)
dat_C = np.loadtxt(dat_C)
x_C, y_C, meas_C, ampl_C, geo = afit.analyze_holography(
    dat_C, tele_geo, 0, 1, 0, shift_C
)
meas_C = np.where(
    (abs(ampl_C) / np.max(abs(ampl_C))) >= 0.3, meas_C - np.mean(meas_C), 0
)

rx1_tri = np.array([rx_x_tri[0], 209.09, rx_z_tri[0]])
tele_geo = tele_geo_init(rx1_tri[0], rx1_tri[1], rx1_tri[2], el_tri[0], az_tri[0])
rxmirror_A_tri = af.ray_mirror_pts(rx1_tri, tele_geo, th, ph)
dat_A = afit.take_measurement(
    np.zeros(77 * 5), np.zeros(77 * 5), 0, tele_geo, rxmirror_A_tri
)
dat_A = np.loadtxt(dat_A)
x_A_tri, y_A_tri, meas_A_tri, ampl_A_tri, geo = afit.analyze_holography(
    dat_A, tele_geo, 0, 1, 0, shift_A_tri
)
meas_A_tri = np.where(
    (abs(ampl_A_tri) / np.max(abs(ampl_A_tri))) >= 0.3,
    meas_A_tri - np.mean(meas_A_tri),
    0,
)

rx2_tri = np.array([rx_x_tri[1], 209.09, rx_z_tri[1]])
tele_geo = tele_geo_init(rx2_tri[0], rx2_tri[1], rx2_tri[2], el_tri[1], az_tri[1])
rxmirror_B_tri = af.ray_mirror_pts(rx2_tri, tele_geo, th, ph)
dat_B = afit.take_measurement(
    np.zeros(77 * 5), np.zeros(77 * 5), 0, tele_geo, rxmirror_B_tri
)
dat_B = np.loadtxt(dat_B)
x_B_tri, y_B_tri, meas_B_tri, ampl_B_tri, geo = afit.analyze_holography(
    dat_B, tele_geo, 0, 1, 0, shift_B_tri
)
meas_B_tri = np.where(
    (abs(ampl_B_tri) / np.max(abs(ampl_B_tri))) >= 0.3,
    meas_B_tri - np.mean(meas_B_tri),
    0,
)

rx3_tri = np.array([rx_x_tri[2], 209.09, rx_z_tri[2]])
tele_geo = tele_geo_init(rx3_tri[0], rx3_tri[1], rx3_tri[2], el_tri[2], az_tri[2])
rxmirror_C_tri = af.ray_mirror_pts(rx3_tri, tele_geo, th, ph)
dat_C = afit.take_measurement(
    np.zeros(77 * 5), np.zeros(77 * 5), 0, tele_geo, rxmirror_C_tri
)
dat_C = np.loadtxt(dat_C)
x_C_tri, y_C_tri, meas_C_tri, ampl_C_tri, geo = afit.analyze_holography(
    dat_C, tele_geo, 0, 1, 0, shift_C_tri
)
meas_C_tri = np.where(
    (abs(ampl_C_tri) / np.max(abs(ampl_C_tri))) >= 0.3,
    meas_C_tri - np.mean(meas_C_tri),
    0,
)


def test():
    adj_err = np.linspace(20, 50, 6)  # Adjuster #1

    trinocular_model = pickle.load(open("../../ml-models/model_trinocular.sav", "rb"))

    initial3 = []
    final3 = []
    final3_m1 = []
    final3_m2 = []

    for ii in range(len(adj_err)):
        print(adj_err[ii])
        # Define FOV of receiver feed (RX) positions
        # (i.e. define direction of outgoing rays from the RX).
        tele_geo = tg.initialize_telescope_geometry()

        # Define adjuster offsets, as random distribution on the micron scale
        adj_1 = np.random.randn(77 * 5) * adj_err[ii]  # [um]
        adj_2 = np.random.randn(69 * 5) * adj_err[ii]  # [um]

        # Define panels on M1 and M2. Here you can define the
        # magnitude of the adjuster offsets on each mirror:
        pan_mod_m2 = pm.panel_model_from_adjuster_offsets(
            2, adj_2, 1, 0
        )  # Panel Model on M2
        pan_mod_m1 = pm.panel_model_from_adjuster_offsets(
            1, adj_1, 1, 0
        )  # Panel Model on M1

        rx1_tri = np.array([rx_x_tri[0], 209.09, rx_z_tri[0]])
        tele_geo = tele_geo_init(
            rx1_tri[0], rx1_tri[1], rx1_tri[2], el_tri[0], az_tri[0]
        )
        out1_tri = af.aperature_fields_from_panel_model(
            pan_mod_m1, pan_mod_m2, rx1_tri, tele_geo, th, ph, rxmirror_A_tri
        )  # apert. fields

        beam1_tri = ff.far_field_sim(out1_tri, tele_geo, rx1_tri)  # far field beam
        amp1_tri = 20 * np.log10(
            abs(beam1_tri[2, :]) / np.max(abs(beam1_tri[2, :]))
        )  # far field beam amplitude [dB]

        rx2_tri = np.array([rx_x_tri[1], 209.09, rx_z_tri[1]])
        tele_geo = tele_geo_init(
            rx2_tri[0], rx2_tri[1], rx2_tri[2], el_tri[1], az_tri[1]
        )
        out2_tri = af.aperature_fields_from_panel_model(
            pan_mod_m1, pan_mod_m2, rx2_tri, tele_geo, th, ph, rxmirror_B_tri
        )  # apert. fields

        beam2_tri = ff.far_field_sim(out2_tri, tele_geo, rx2_tri)  # far field beam
        amp2_tri = 20 * np.log10(
            abs(beam2_tri[2, :]) / np.max(abs(beam2_tri[2, :]))
        )  # far field beam amplitude [dB]

        rx3_tri = np.array([rx_x_tri[2], 209.09, rx_z_tri[2]])
        tele_geo = tele_geo_init(
            rx3_tri[0], rx3_tri[1], rx3_tri[2], el_tri[2], az_tri[2]
        )
        out3_tri = af.aperature_fields_from_panel_model(
            pan_mod_m1, pan_mod_m2, rx3_tri, tele_geo, th, ph, rxmirror_C_tri
        )  # apert. fields

        beam3_tri = ff.far_field_sim(out3_tri, tele_geo, rx3_tri)  # far field beam
        amp3_tri = 20 * np.log10(
            abs(beam3_tri[2, :]) / np.max(abs(beam3_tri[2, :]))
        )  # far field beam amplitude [dB]

        amp3_tri = 20 * np.log10(
            abs(beam3_tri[2, :]) / np.max(abs(beam3_tri[2, :]))
        )  # far field beam amplitude [dB]

        np.savetxt(
            "/data/chesmore/sim_out/rx_" + str(rx1_tri) + "_holog_tri.txt",
            np.c_[
                np.real(beam1_tri[0, :]),
                np.real(beam1_tri[1, :]),
                np.real(beam1_tri[2, :]),
                np.imag(beam1_tri[2, :]),
            ],
        )
        np.savetxt(
            "/data/chesmore/sim_out/rx_" + str(rx2_tri) + "_holog_tri.txt",
            np.c_[
                np.real(beam2_tri[0, :]),
                np.real(beam2_tri[1, :]),
                np.real(beam2_tri[2, :]),
                np.imag(beam2_tri[2, :]),
            ],
        )
        np.savetxt(
            "/data/chesmore/sim_out/rx_" + str(rx3_tri) + "_holog_tri.txt",
            np.c_[
                np.real(beam3_tri[0, :]),
                np.real(beam3_tri[1, :]),
                np.real(beam3_tri[2, :]),
                np.imag(beam3_tri[2, :]),
            ],
        )

        tele_geo = tele_geo_init(
            rx1_tri[0], rx1_tri[1], rx1_tri[2], el_tri[0], az_tri[0]
        )
        dat_A = np.loadtxt(
            "/data/chesmore/sim_out/rx_" + str(rx1_tri) + "_holog_tri.txt"
        )
        x_A_tri, y_A_tri, phase_A_tri, ampl_A_tri, geo = afit.analyze_holography(
            dat_A, tele_geo, 0, 1, 0, shift_A_tri
        )

        tele_geo = tele_geo_init(
            rx2_tri[0], rx2_tri[1], rx2_tri[2], el_tri[1], az_tri[1]
        )
        dat_B = np.loadtxt(
            "/data/chesmore/sim_out/rx_" + str(rx2_tri) + "_holog_tri.txt"
        )
        x_B_tri, y_B_tri, phase_B_tri, ampl_B_tri, geo = afit.analyze_holography(
            dat_B, tele_geo, 0, 1, 0, shift_B_tri
        )

        tele_geo = tele_geo_init(
            rx3_tri[0], rx3_tri[1], rx3_tri[2], el_tri[2], az_tri[2]
        )
        dat_C = np.loadtxt(
            "/data/chesmore/sim_out/rx_" + str(rx3_tri) + "_holog_tri.txt"
        )
        x_C_tri, y_C_tri, phase_C_tri, ampl_C_tri, geo = afit.analyze_holography(
            dat_C, tele_geo, 0, 1, 0, shift_C_tri
        )

        phase_A_tri = np.where(
            (abs(ampl_A_tri) / np.max(abs(ampl_A_tri))) >= 0.3,
            phase_A_tri - np.mean(phase_A_tri),
            0,
        )
        phase_B_tri = np.where(
            (abs(ampl_B_tri) / np.max(abs(ampl_B_tri))) >= 0.3,
            phase_B_tri - np.mean(phase_B_tri),
            0,
        )
        phase_C_tri = np.where(
            (abs(ampl_C_tri) / np.max(abs(ampl_C_tri))) >= 0.3,
            phase_C_tri - np.mean(phase_C_tri),
            0,
        )
        phase_A_tri -= meas_A_tri
        phase_B_tri -= meas_B_tri
        phase_C_tri -= meas_C_tri

        pathl_meas3 = np.reshape(
            (np.concatenate((phase_A_tri, phase_B_tri, phase_C_tri))),
            (1, len(np.concatenate((phase_A_tri, phase_B_tri, phase_C_tri)))),
        )
        adj_fit3 = trinocular_model.predict(pathl_meas3)

        adjs_real = np.concatenate((adj_1, adj_2)) / 1e3

        adjust = adj_fit3[0]

        rx3 = np.array([rx_x[2], 209.09, rx_z[2]])
        tele_geo_C = tele_geo_init(rx3[0], rx3[1], rx3[2], el[2], az[2])
        phase_C = af.model_of_adj_offs(adjs_real * 1e3, shift_C, tele_geo_C, "total")
        phase_tot_new_C3 = af.model_of_adj_offs(
            ((adjs_real - adjust) * 1e3), shift_C, tele_geo_C, "total"
        )
        phase_m1_new_C3 = af.model_of_adj_offs(
            ((adjs_real - adjust) * 1e3), shift_C, tele_geo_C, "m1"
        )
        phase_m2_new_C3 = af.model_of_adj_offs(
            ((adjs_real - adjust) * 1e3), shift_C, tele_geo_C, "m2"
        )

        final3_m1.append(
            oa.rms(
                x_C,
                y_C,
                1e6
                * (
                    phase_m1_new_C3
                    - np.mean(phase_m1_new_C3)
                    - (meas_C - np.mean(meas_C))
                )
                / tele_geo.k,
            )
        )
        final3_m2.append(
            oa.rms(
                x_C,
                y_C,
                1e6
                * (
                    phase_m2_new_C3
                    - np.mean(phase_m2_new_C3)
                    - (meas_C - np.mean(meas_C))
                )
                / tele_geo.k,
            )
        )

        initial3.append(
            oa.rms(
                x_C,
                y_C,
                1e6
                * (phase_C - np.mean(phase_C) - (meas_C - np.mean(meas_C)))
                / tele_geo.k,
            )
        )
        final3.append(
            oa.rms(
                x_C,
                y_C,
                1e6
                * (
                    phase_tot_new_C3
                    - np.mean(phase_tot_new_C3)
                    - (meas_C - np.mean(meas_C))
                )
                / tele_geo.k,
            )
        )

        adj_err_final = adj_err[ii]
        print(final3)

        if ii == 0:
            val_init = oa.rms(
                x_C,
                y_C,
                1e6
                * (
                    phase_tot_new_C3
                    - np.mean(phase_tot_new_C3)
                    - (meas_C - np.mean(meas_C))
                )
                / tele_geo.k,
            )

        if (
            oa.rms(
                x_C,
                y_C,
                1e6
                * (
                    phase_tot_new_C3
                    - np.mean(phase_tot_new_C3)
                    - (meas_C - np.mean(meas_C))
                )
                / tele_geo.k,
            )
            > (val_init + 5)
        ):
            break

    return adj_err_final


values = []
for ii in range(10):

    print(str(ii + 1) + "/10")
    value = test()
    values.append(value)

print(np.mean(values), np.std(values))
