# HH_bbWW

## How to run anaTuple production

Current version `v2605`

1. Clone repository
   ```bash
   git clone -b v2605 --recursive git@github.com:cms-flaf/HH_bbWW.git
   cd HH_bbWW
   git lfs pull
   source $PWD/env.sh
   law index
   ```

1. Define `config/user_custom.yaml` file as following:
   ```yaml
   fs_default: T3_CH_CERNBOX:/store/user/YOUR_USER_NAME/HH_bbWW/
   fs_anaTuple: T3_US_FNALLPC:/store/user/lpcflaf/HH_bbWW/

   analysis_config_area: config
   compute_unc_variations: true
   compute_unc_histograms: true
   store_noncentral: true
   ```

1. Login to cms-flaf.cern.ch, enter screen session, login to lxplus
   ```bash
   ssh USER@cms-flaf.cern.ch
   screen -S HH_bbWW_production
   ssh lxplus.cern.ch
   ```

1. Load environment and setup grid certificate
   ```bash
   source $PWD/env.sh
   voms-proxy-init --voms cms --valid 192:00
   ```

1. Run production
   ```bash
   law run AnaTupleMergeTask --version v2605 --period ERA --parallel-jobs 1000 --AnaTupleFileTask-tasks-per-job 10
   ```
