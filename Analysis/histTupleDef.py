import importlib
from FLAF.Common.Utilities import *
from FLAF.Common.HistHelper import *
from Corrections.Corrections import Corrections
from Corrections.CorrectionsCore import getSystName, central

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

initialized = False
analysis = None


def Initialize():
    global initialized
    if not initialized:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')
        initialized = True


def analysis_setup(setup):
    global analysis
    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")


def GetDfw(
    df,
    df_caches,
    global_params,
    shift="Central",
    col_names_central=[],
    col_types_central=[],
    cache_map_name="cache_map_Central",
):
    period = global_params["era"]
    kwargset = (
        {}
    )  # here go the customisations for each analysis eventually extrcting stuff from the global params
    # Example from Hmm analysis:

    # kwargset["isData"] = global_params["process_group"] == "data"
    # kwargset["wantTriggerSFErrors"] = global_params["compute_rel_weights"]
    # kwargset["colToSave"] = []

    dfw = analysis.DataFrameBuilderForHistograms(df, global_params, period, **kwargset)

    if df_caches:
        k = 0
        for df_cache in df_caches:
            dfWrapped_cache = analysis.DataFrameBuilderForHistograms(
                df_cache, global_params, period, **kwargset
            )
            AddCacheColumnsInDf(dfw, dfWrapped_cache, f"{cache_map_name}_{k}")
            k += 1

    if shift == "Valid" and global_params["compute_unc_variations"]:
        dfw.CreateFromDelta(col_names_central, col_types_central)
    if shift != "Central" and global_params["compute_unc_variations"]:
        dfw.AddMissingColumns(col_names_central, col_types_central)
    new_dfw = analysis.PrepareDfForHistograms(dfw)
    return new_dfw


central_df_weights_computed = False


def DefineWeightForHistograms(
    *,
    dfw,
    isData,
    uncName,
    uncScale,
    unc_cfg_dict,
    hist_cfg_dict,
    global_params,
    final_weight_name,
    df_is_central,
):
    global central_df_weights_computed
    if not isData and (not central_df_weights_computed or not df_is_central):
        corrections = Corrections.getGlobal()
        lepton_legs = ["lep1", "lep2"]
        offline_legs = ["lep1", "lep2"]
        triggers_to_use = set()
        channels = global_params["channelSelection"]
        for channel in channels:
            trigger_list = global_params.get("triggers", {}).get(channel, [])
            for trigger in trigger_list:
                if trigger not in corrections.trigger_dict.keys():
                    raise RuntimeError(
                        f"Trigger does not exist in triggers.yaml, {trigger}"
                    )
                triggers_to_use.add(trigger)
        syst_name = getSystName(uncName, uncScale)
        is_central = uncName == central

        dfw.df, all_weights = corrections.getNormalisationCorrections(
            dfw.df,
            lepton_legs=lepton_legs,
            offline_legs=offline_legs,
            trigger_names=triggers_to_use,
            syst_name=syst_name,
            source_name=uncName,
            ana_caches=None,
            return_variations=is_central and global_params["compute_unc_histograms"],
            isCentral=is_central,
            use_genWeight_sign_only=True,
        )
        if df_is_central:
            central_df_weights_computed = True

    categories = global_params["categories"]
    boosted_categories = global_params.get("boosted_categories", [])
    process_group = global_params["process_group"]
    isCentral = uncName == "Central"
    total_weight_expression = (
        # channel, cat, boosted_categories --> these are not needed in the GetWeight function therefore I just put some placeholders
        analysis.GetWeight("", "", boosted_categories)
        if process_group != "data"
        else "1"
    )  # are we sure?
    weight_name = "final_weight"
    if weight_name not in dfw.df.GetColumnNames():
        dfw.df = dfw.df.Define(weight_name, total_weight_expression)
    if not isCentral and type(unc_cfg_dict) == dict:
        if (
            uncName in unc_cfg_dict["norm"].keys()
            and "expression" in unc_cfg_dict["norm"][uncName].keys()
        ):
            weight_name = unc_cfg_dict["norm"][uncName]["expression"].format(
                scale=uncScale
            )
    dfw.df = dfw.df.Define(final_weight_name, weight_name)
