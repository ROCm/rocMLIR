#!/bin/bash

# For installing mysql 8.0 for testing, or for running with an isolated database.
function mysql_setup_generic
{
    # Note that all this happens without privileges.
    export PATH=$PATH:/usr/mysql/bin
    mysqld --initialize-insecure --datadir=/tmp/mysql-data
    mysqld -D --basedir=/usr/mysql --datadir=/tmp/mysql-data --log-error=/tmp/mysql-errors.log --secure-file-priv=''

    # Using this name should force socket access, which we want.
    TUNA_DB_HOSTNAME=localhost
    mysql --user root -e 'create database tuna;'
}

function tuna_setup
{
#     rm -rf /tmp/MITuna
#     git clone --branch pf-tuna-rocmlir-3 http://github.com/ROCmSoftwarePlatform/MITuna.git /tmp/MITuna

    source /tuna-venv/bin/activate
    #export TUNA_DIR=/tmp/MITuna
    export PYTHONPATH=$TUNA_DIR:$PYTHONPATH

    if pgrep mysqld ; then
        ${TUNA_DIR}/tuna/go_fish.py rocmlir --add_tables
    fi
}

function clear_tables
{
    tablekind=$1

    if [ "$tablekind" = "convolution" ]; then
        tablekind="conv"
    fi

    # config table has foreign keys from job and results tables.  however,
    # config table doesn't have a session column so we must delete them all.
    mysql --user root --database tuna -e "delete from rocmlir_${tablekind}_results;"
    mysql --user root --database tuna -e "delete from rocmlir_${tablekind}_job;"
    mysql --user root --database tuna -e "delete from rocmlir_${tablekind}_config;"
}

function tuna_run
{
    kind=$1
    space=$2
    baselabel=$(date --iso-8601=minutes)

    clear_tables "$kind"
    ${TUNA_DIR}/tuna/rocmlir/import_configs.py --file_name "${CONFIGS_FILE}" --config_type "$kind"
    ${TUNA_DIR}/tuna/go_fish.py rocmlir --init_session -l "$baselabel $kind" --config_type "$kind" --tuning_space "$space" 2> initlog
    session=$(perl -n -e'/Added new session_id: (\d+)/ && print $1' < initlog)
    cat initlog
    ${TUNA_DIR}/tuna/rocmlir/load_job.py --session_id "$session"
    factor=""
    if [ -n "${LOAD_FACTOR}" ]; then
        factor="--load_factor ${LOAD_FACTOR}"
    fi
    (cd "${ROCMLIR_DIR}"/build/ || exit 1 ; ${TUNA_DIR}/tuna/go_fish.py rocmlir --execute --session_id "$session" $factor)
    ${TUNA_DIR}/tuna/rocmlir/export_configs.py --session_id "$session" --append -f "$OUT_FILE" $CSV
}



usage() { echo "$0 usage:" && grep " .)\ #" "$0"; exit 0; }
[ $# -eq 0 ] && usage

export CONFIGS_FILE=
export TUNA_DIR=/tmp/MITuna
export ROCMLIR_DIR=$(pwd)/..      # Assumes we're in the build directory
export OUT_FILE=results.tsv
export OP=convolution
export TUNING_SPACE=exhaustive
export LOAD_FACTOR=
export CSV=

# -c configs
# -t tunadir
# -r rocmlirdir
# -f outfile
# -o operation
# -s tuning space
# -l load factor
while getopts ":hc:t:r:f:o:s:l:u" arg; do
  case $arg in
    o) # Operation (convolution or gemm [default convolution])
      OP=${OPTARG}
      [ "$OP" = "convolution" ] || [ "$OP" = "gemm" ] || [ "$OP" = "attention" ] \
        || echo "Operation needs to be 'convolution', 'gemm', or 'attention'."
      ;;
    c) # Configs file
      CONFIGS_FILE="${OPTARG}"
      ;;
    t) # Location of existing Tuna installation
      TUNA_DIR="${OPTARG}"
      ;;
    r) # Location of rocMLIR
      ROCMLIR_DIR="${OPTARG}"
      ;;
    f) # File to write tuning results to.
      OUT_FILE="${OPTARG}"
      ;;
    s) # Tuning space (default exhaustive)
      TUNING_SPACE="${OPTARG}"
      ;;
    l) # Load factor (default 1.0)
      LOAD_FACTOR="${OPTARG}"
      ;;
    u) # Also export as .csv for upload.
      CSV="--csv"
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done

export TUNA_DB_USER_NAME=root
export TUNA_DB_USER_PASSWORD=
export TUNA_DB_HOSTNAME=127.0.0.1
export TUNA_DB_NAME=tuna
export PYTHONPATH=$TUNA_DIR:$PYTHONPATH

# If no mysqld running, assume it and Tuna need to be set up.
# Otherwise, assume the usual setup.
if ! pgrep mysqld ; then
    mysql_setup_generic
    tuna_setup
else
    PATH=$PATH:/usr/mysql/bin
    TUNA_DB_HOSTNAME=localhost
fi

if [ "$VIRTUAL_ENV" = "" ]; then
    source /tuna-venv/bin/activate
fi

tuna_run "$OP" "$TUNING_SPACE"
