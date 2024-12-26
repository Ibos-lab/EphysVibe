from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.dataviz import plot_raster
from pathlib import Path
import matplotlib.pyplot as plt
import os


def prepare_and_plot(
    neupath: Path,
    format: str = "png",
    percentile: bool = False,
    cerotr: bool = False,
    b: int = 1,
    output_dir: Path = "./",
):

    neu = NeuronData.from_python_hdf5(neupath)
    nid = neu.get_neuron_id()
    print(nid)
    if b == 1:
        sp, conv = plot_raster.prepare_data_plotb1(
            neu,
            rf_stim_loc=["contra", "ipsi"],
            percentile=percentile,
            cerotr=cerotr,
        )

        fig = plot_raster.plot_sp_b1(neu, sp, conv)

    elif b == 2:
        sp_pos, conv_pos, max_n_tr, conv_max = plot_raster.prepare_data_plotb2(neu)
        fig = plot_raster.plot_sp_b2(
            neu, sp_pos, conv_pos, max_n_tr, conv_max, visual_rf=True, inout=1
        )

    fig.savefig(
        f"{output_dir}/{nid}.{format}",
        format=format,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close(fig)
