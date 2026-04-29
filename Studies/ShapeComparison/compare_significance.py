import uproot
import numpy as np
import os


def main():
    category = "resolved2b"
    year = 2018

    masspoints = [600]
    # backgrounds = ["TT", "DY", "ST", "Fakes"]
    backgrounds = ["m{masspoint}_TT", "m{masspoint}_DY", "m{masspoint}_Other"]

    # percentage
    signif_inc_thresh = 1

    for mp in masspoints:
        # file_name = f"HH_DL_{mp}_{category}_GGF_{year}.root"
        file_name = f"run3_res2b.root"
        file_name = f"run3_res2b_11mar.root"
        file_name = f"run3_res2b_mar16.root"

        f = uproot.open(file_name)

        # sig_hist = f[f"signal_ggf_spin0_{mp}_hbbhww"]
        sig_hist = f[f"m{mp}_Signal"]

        sig_counts, _ = sig_hist.to_numpy()

        n_bins = len(sig_counts)
        total_bkg_counts = np.zeros(n_bins)
        bkg_counts_dict = {}
        for bkg in backgrounds:
            bkg = bkg.format(masspoint=mp)

            bkg_hist = f[bkg]
            bkg_counts, _ = bkg_hist.to_numpy()
            bkg_counts_dict[bkg] = bkg_counts
            total_bkg_counts += bkg_counts

        print(
            f"Full shape has nSignal: {np.sum(sig_counts)} -- and nBackground: {np.sum(total_bkg_counts)}"
        )
        print(
            f"Shape[30:] has nSignal: {np.sum(sig_counts[30:])} -- and nBackground: {np.sum(total_bkg_counts[30:])}"
        )
        print(
            f"Shape[23:] has nSignal: {np.sum(sig_counts[23:])} -- and nBackground: {np.sum(total_bkg_counts[23:])}"
        )

        # bin_significances = sig_counts/np.sqrt(total_bkg_counts)
        bin_significances = sig_counts / np.sqrt(total_bkg_counts + sig_counts)
        sorted_bin_indices = np.argsort(bin_significances)[::-1]

        sig_cnt = sig_counts[sorted_bin_indices[0]]
        bkg_cnt = total_bkg_counts[sorted_bin_indices[0]]
        # signif = sig_cnt/np.sqrt(bkg_cnt)
        signif = sig_cnt / np.sqrt(bkg_cnt + sig_cnt)
        bins = [sorted_bin_indices[0]]
        print(f"Most significant bin is {sorted_bin_indices[0]}")
        for bin_idx in sorted_bin_indices[1:]:
            prev_signif = signif
            print(f"Considering bin {bin_idx}")
            print(
                f"\tBefore: sig_cnt={sig_cnt:.3f}, bkg_cnt={bkg_cnt:.3f}, signif={signif:.3f}"
            )
            sig_cnt += sig_counts[bin_idx]
            bkg_cnt += total_bkg_counts[bin_idx]
            # signif = sig_cnt/np.sqrt(bkg_cnt)
            signif = sig_cnt / np.sqrt(bkg_cnt + sig_cnt)
            print(
                f"\tAfter: sig_cnt={sig_cnt:.3f}, bkg_cnt={bkg_cnt:.3f}, signif={signif:.3f}"
            )
            gain = (signif - prev_signif) / prev_signif * 100
            print(f"\tGain: {gain:.2f}%")
            if gain >= signif_inc_thresh:
                bins.append(bin_idx)
            else:
                break

        print(f"Selected {len(bins)} out of {n_bins} bins")
        print(f"Suggest merging bins {np.sort(bins)}")

        bkg_fractions = {}
        tot_sig_selected = np.sum(sig_counts[bins])
        tot_bkg_selected = np.sum(total_bkg_counts[bins])
        for bkg, bkg_counts in bkg_counts_dict.items():
            bkg_fractions[bkg] = np.sum(bkg_counts[bins]) / tot_bkg_selected

        print("Obtained background fractions:")
        for bkg, frac in bkg_fractions.items():
            print(f"{bkg}: {frac*100:.2f}%")

        print("Coming from initial bkg/sig dicts")
        print(bkg_counts_dict)
        print(sig_counts)

        print("All significances")
        print(bin_significances)


if __name__ == "__main__":
    main()
