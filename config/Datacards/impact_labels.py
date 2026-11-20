"""dhi `labels` file of the impact plots (DL and SL; common.yaml impact_plots plot_params): shortens the bin-by-bin
MC-statistics nuisance names, for display only, so the AN layout's label margin fits them.

prop_binRun3_22_23_hh_bbww_<channel>_<region>_<category>_bin<N>[_<process>] -> prop_bin_<channel>_<region>_<category>_bin<N>[_<process>],
with the signal processes shortened (GluGluToRadion_bbWW_2L_300 -> sig_bbWW_2L). Other names are unchanged (None).
dhi imports this file and calls rename_nuisance(name) for every parameter on the plot.
"""

import re


def rename_nuisance(name):
    if not name.startswith("prop_binRun3_22_23_hh_bbww_"):
        return None
    short = "prop_bin_" + name[len("prop_binRun3_22_23_hh_bbww_") :]
    return re.sub(r"_GluGluTo(?:Radion|BulkGraviton)_(\w+?)_\d+$", r"_sig_\1", short)
