# ShapeYields

What is actually in the bins combine is fitting.

`shape_yields.py` reads the rebinned shapes -- the output of the datacard configuration's
`preprocess:` step, which for HH->bbWW is `StatInference/bin_opt_2d/rebin_2d.py`, and which
is the same thing `CreateDatacardsTask` reads as its input -- and reports every bin's
content and MC-statistical error, per process, per channel, per era.

Nominal shapes only. The Up/Down variations are in the same files and are left alone.

## Running it

```
python3 Studies/ShapeYields/shape_yields.py \
    --input /eos/user/d/daebi/HH_bbWW/<version>/Hists_preprocessed/Run3_Early \
    --config config/Datacards/x_hh_bbww_DL_run3.yaml \
    --binning-config config/Datacards/binning_2d.yaml \
    --output Studies/ShapeYields/output/<version>
```

`--input` is the era-group directory, holding one sub-directory per source era. The four
sub-eras are read individually and their sum is reported alongside them as `Run3_Early`,
which is the thing the datacards are built from.

`--era`, `--mass`, `--channel` and `--category` are all repeatable and all narrow the run.
Give them to cut a single page for a slide instead of the whole set:

```
    --era Run3_Early --mass 500 --channel muMu --category SR/res2b_dnn1
```

`--no-pdf` writes only the CSV.

## What comes out

`yields.csv` -- one row per (era, mass, channel, category, bin, process), carrying the
bin's HME edges, its content and its error. Tidy, so it sorts and greps and reads into
pandas without reshaping.

`yields_<era>.pdf` -- the same numbers as tables, one page per
(era, mass, channel, base category) with the four DNN slices stacked down the page. Each
page is a slide. Cell shade is log magnitude within the slice; negative content is boxed
and printed in red.

## The uncv2 run

Kept beside the shapes it was read from:

```
/eos/user/d/daebi/HH_bbWW/uncv2/ShapeYields/
```

`yields.csv` (54,995 rows) and one 87-page PDF per era -- the four sub-eras and their sum.

## Checking it

`verify_against_datacards.py` compares the CSV against the datacard shapes
`CreateDatacardsTask` wrote, in both directions: every nominal histogram in the datacard
must be reproducible from the CSV, and every background in the CSV must be consumed by the
datacard. It exits non-zero on any unmatched histogram or any bin difference above
tolerance.

```
python3 Studies/ShapeYields/verify_against_datacards.py \
    --csv Studies/ShapeYields/output/uncv2/yields.csv \
    --datacards data/uncv2/Datacards/Run3_Early \
    --config config/Datacards/x_hh_bbww_DL_run3.yaml \
    --era Run3_Early
```

On uncv2: 1684 histograms over all ten masses, worst relative bin difference 3.9e-16,
nothing unmatched in either direction.

Both directions matter. Checking only "is every datacard histogram reproducible" lets the
merged `TotalBkg` categories be skipped as explained by the merge, and that is exactly
where a real defect hid -- see the note in `merged_scopes()`.

What the check does and does not establish: it establishes that this tool reads and sums
the shapes the way the datacard chain does. It cannot establish that the shapes are
physically right, since both sides read the same files -- an error upstream in the
rebinning would be reproduced identically by both.

## What it does not do

It applies no thresholds and passes no judgement on which bins are acceptable. It does not
consult the binning that produced the shapes, and it re-derives nothing from
`rebin_2d.py` -- the numbers in the files are the report, and the reading is meant to be
independent of the code that wrote them.

Everything but the file layout comes from the datacard configuration: the processes and
their channel/category restrictions, the masses, the channels, the categories, and the
`input_file_pattern` the paths are built from. Two kinds of process are deliberately
absent. `data_obs` is Asimov (`is_asimov_data`), so it carries nothing the summed
background does not, and the `Total bkg` row is that sum. `TotalBkg` -- the merged boosted
template -- is assembled by the datacard maker and is not a histogram in these files; its
four constituents are, and they are reported individually, so the boosted pages show what
went into the merge rather than the merge.

Signal rows are the raw histograms, before the `scale` the datacard applies to unfold the
H->bb and H->WW branching fractions. They will not match a datacard `rate` line for that
reason; the background rows will.
