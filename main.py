from GPR import run_gpr
from PLOT import plot_all


def main():
    result = run_gpr(
        stl_filepath="input_stls/triangle.stl",
        cfd_filepath="inputs/FLTG.csv",
        stl_scale=1.0 / 1000.0,
        training_point_n_requested=80,
        res=30,
        v_inf=(12.0, 0.0, 0.0),
        n_restarts=6,
        fit_pressure=True,
    )

    plot_all(result, z_slice_target=2.5, show=True)


if __name__ == "__main__":
    main()