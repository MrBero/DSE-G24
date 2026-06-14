from GPR import run_gpr
from PLOT import plot_posterior_3d, plot_slice_comparison, plot_multi_slices, plot_pressure_slice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def main():
    result = run_gpr(
        stl_filepath="input_stls/wind_turbine3_fixed.stl",
        cfd_filepath="inputs/CSV_WIND_TURBINE_5X.pkl",
        prior_means_filepath='inputs/CSV_INVISCID_WIND_TURBINE_5X.pkl',
        stl_scale=1.0 / 1000.0,
        # stl_rotate=-np.pi/2,
        res=50,
        v_inf=(0.0, 12.0, 0.0),
        bounds_input=np.array([[-100,100],[-175,125],[0,175]]),
        n_restarts=6,
        fit_pressure=True,
        posterior_batch=100,
        #random
        # sample_method='random',
        # num_samples=120
        #Cylinder
        sample_method="cylinder",
        sample_config={"r_factor": 0.7, "h_factor": 2, "tilt_deg": 5,
                    "n_points": 400, "front_frac": 0.15, "front_half_angle_deg": 45}
        # Drone array
        # sample_method="drone_array",
        # sample_config={"tilt_deg": 30, "n_rows": 10, "n_cols": 10},

    )

    figs = [
        plot_posterior_3d(result, field_alpha=0.0),
        plot_slice_comparison(result, z_slice_target=45),
        plot_multi_slices(result, n_slices=5, field="posterior", axis="z", slice_range=(5,50)),
        plot_multi_slices(result, n_slices=5, field="variances", axis="z", slice_range=(5,50)),
        plot_multi_slices(result, n_slices=5, field="posterior", axis="y", slice_range=(-10,40)),
        plot_multi_slices(result, n_slices=5, field="variances", axis="y", slice_range=(-10,40)),
        # plot_multi_slices(result, n_slices=n_slices, field="variances", axis="z")
        plot_pressure_slice(result, z_slice_target=45)
    ]
    multipage('plots/wind_turbine_results.pdf', figs = figs)
    plt.show()

if __name__ == "__main__":
    main()