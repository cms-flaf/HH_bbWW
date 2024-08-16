import argparse
from PurityChecker import PurityChecker
from ReportUtils import SaveToFile


prefix = "/eos/user/a/abolshov/Analysis/FLAF/anaTuples/hme_v1/Run3_2022"
nano = "nano_0.root"

resonances = ["Graviton", "Radion"]
channels = {"2B2L2Nu":"dl", "2B2JLNu":"sl"}


def main():
    parser = argparse.ArgumentParser(prog='check_all', description='Checks purity of the given sample')
    parser.add_argument('file_name', type=str, help="File with list of input files")
    parser.add_argument('sort_by', type=str, help="Name of the branch used to sort and select jets")

    args = parser.parse_args()

    cfg_name = args.file_name
    tagger_name = args.sort_by

    with open(cfg_name, 'r') as cfg:
        paths = [f"{prefix}/{line[:-1]}/{nano}" for line in cfg.readlines()]

        pc = PurityChecker(tagger_name)

        for res in resonances:
            for channel_id, channel_name in channels.items():
                purity = [pc.ComputePurity(p, channel_name) for p in paths if channel_id in p and res in p]
                masspoint = [float(p.split('-')[-1].split('/')[0]) for p in paths if channel_id in p and res in p]
                SaveToFile(f"out_{channel_name}_{res}_{tagger_name.split('-')[-1]}.txt", masspoint, purity)


if __name__ == '__main__':
    main()