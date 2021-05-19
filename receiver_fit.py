import numpy as np
import tele_geo as tg
import ap_field as af
import pan_mod as pm
import optics_analyze as oa
import far_field as ff
import ap_fitting as afit
import scipy

# Singular and Binocular positions
rx_x = np.array([0, 0, 0])
rx_z = np.array([0, 0, 0])
el = np.array([0, 0, 0])
az = np.array([0, 0, 0])

shift = ["y", oa.sh_z(rx_z[2])]

# Define FOV of receiver feed (RX) positions
# (i.e. define direction of outgoing rays from the RX).
tele_geo = tg.initialize_telescope_geometry()
th = np.linspace(-np.pi / 2 - 0.28, -np.pi / 2 + 0.28, tele_geo.N_scan)
ph = np.linspace(np.pi / 2 - 0.28, np.pi / 2 + 0.28, tele_geo.N_scan)

# Here we simulate the actual holography measurement,
# which we will later fit to determine its position.
rx1 = np.array([rx_x[2], 209.09, rx_z[2]])
tele_geo = tg.tele_geo_init(rx1[0], rx1[1], rx1[2], el[2], az[2])
rxmirror_A = af.ray_mirror_pts(rx1, tele_geo, th, ph)
dat_A = afit.take_measurement(
    np.zeros(77 * 5), np.zeros(77 * 5), 0, tele_geo, rxmirror_A
)
dat_A = np.loadtxt(dat_A)
x_A, y_A, meas_A, ampl_A, geo = afit.analyze_holography(dat_A, tele_geo, 0, 1, 0, shift)
meas_A = np.where(
    (abs(ampl_A) / np.max(abs(ampl_A))) >= 0.3, meas_A - np.mean(meas_A), 0
)


def fit_rec_chi(p, data, x_A, y_A):
    """
    Minimize this function to determine the
    location of receiver feed in the focal plane.
    """
    rx_mod = np.array([rx_x[2], p[0], rx_z[2]])
    tele_geo_mod = tg.tele_geo_init(rx_mod[0], rx_mod[1], rx_mod[2], el[2], az[2])
    rxmirror_mod = af.ray_mirror_pts(rx_mod, tele_geo_mod, th, ph)
    dat_mod = afit.take_measurement(
        np.zeros(77 * 5), np.zeros(77 * 5), 0, tele_geo_mod, rxmirror_mod
    )
    dat_mod = np.loadtxt(dat_mod)
    x_mod, y_mod, meas_mod, ampl_mod, geo = afit.analyze_holography(
        dat_mod, tele_geo_mod, 0, 1, 0, shift
    )
    meas_mod = np.where(
        (abs(ampl_mod) / np.max(abs(ampl_mod))) >= 0.3, meas_mod - np.mean(meas_mod), 0
    )

    if len(meas_mod) < len(data):

        return np.sum(
            np.sqrt(
                (meas_mod[0 : len(meas_mod) - 1] - data[0 : len(meas_mod) - 1]) ** 2
            )
        )
    elif len(meas_mod) >= len(data):

        return np.sum(
            np.sqrt((meas_mod[0 : len(data) - 1] - data[0 : len(data) - 1]) ** 2)
        )


p = np.array([(220)])  # receiver position guess [mm]
bnds = [(p[0] - 20, p[0] + 20) for _ in p]
val_y = scipy.optimize.minimize(
    fit_rec_chi,
    x0=p,
    args=(meas_A, x_A, y_A),
    method="L-BFGS-B",
    bounds=bnds,
    options={"eps": 0.1, "maxfun": 20, "maxiter": 20},
)
