action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    local this_file_path="$this_dir/$(basename $this_file)"

    export ANALYSIS_PATH="$this_dir"
    export HH_INFERENCE_PATH="$ANALYSIS_PATH/inference"
    # FLAF_PATH defaults to the submodule copy but is respected if pre-set (flaf_dev.sh
    # points it at the edited top-level FLAF in a FLAF_all workspace).
    [ -z "$FLAF_PATH" ] && export FLAF_PATH="$ANALYSIS_PATH/FLAF"

    source "$FLAF_PATH/env.sh" "$this_file_path" "$@"
}

action "$@"
unset -f action
