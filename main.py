from GPR import run_gpr
from PLOT import plot_all
import numpy as np


def main():
    result = run_gpr(
        stl_filepath="input_stls/Aerospecial_building.stl",
        cfd_filepath="inputs/csv_with_everything.pkl",
        stl_scale=1.0 / 1000.0,
        stl_rotate=-np.pi/2,
        training_point_n_requested=200,
        method='random',
        res=30,
        v_inf=(0.0, 12.0, 0.0),
        n_restarts=6,
        fit_pressure=True,
        verbose=False,

        sample_method="cylinder",
        sample_config={"r_factor": 0.8, "h_factor": 1.5, "tilt_deg": 30,
                    "n_points": 90, "front_frac": 0.5, "front_half_angle_deg": 45},

        # Drone array
        # sample_method="drone_array",
        # sample_config={"tilt_deg": 30, "n_rows": 10, "n_cols": 10},

    )

    plot_all(result, z_slice_target=5, show=True)


if __name__ == "__main__":
    main()