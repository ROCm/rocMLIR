#!/bin/bash

set -x

# For installing mysql 8.0 for testing, or for running with an isolated database.
function mysql_setup
{
    # Fetch mysql 8.0 specifically, since it isn't the latest in ubuntu repos.
    wget https://dev.mysql.com/get/mysql-apt-config_0.8.26-1_all.deb

    # Install.  The .deb sets up the apt paths.
    export DEBIAN_FRONTEND=noninteractive
    dpkg -i ./mysql-apt-config_0.8.26-1_all.deb
    apt update
    # set +e
    apt install -y mysql-community-server
    if [ `/sbin/runlevel` = 'unknown' ]; then
        sudo -u mysql mysqld -D  # -P 13306        # Specific to under docker.
    fi

    # Initial mysql has no root password.  One must be root to run it.  Make a root
    # password and allow password logins, and create our tuna database.
    mysql <<FOO
alter user 'root'@'localhost' identified with mysql_native_password by 'TunaTest';
flush privileges;
create database tuna;
FOO

    cat <<EOF >> $HOME/.my.cnf
[client]
user = root
password = TunaTest
EOF
}


function tuna_setup
{
    git clone --branch pf-tuna-rocmlir-2 https://github.com/ROCmSoftwarePlatform/MITuna.git
    cd MITuna
    export TUNA_DIR=`pwd`

    # +++pf: skip venv when inside docker?
    #if [ `/sbin/runlevel` != 'unknown' ]; then
        python3 -m venv myvenv
        source myvenv/bin/activate
    #fi
    # --ignore-installed because of problems upgrading PyYAML.  See also -U.
    python3 -m pip install -r requirements.txt --ignore-installed
    python3 -m pip install scipy pandas
}


function tuna_populate
{
    ${TUNA_DIR}/tuna/go_fish.py rocmlir --add_tables
    ${TUNA_DIR}/tuna/rocmlir/import_configs.py --file_name ${ROCMLIR_DIR}/mlir/utils/jenkins/ci-configs/selected-conv-configs
                                                           #${ROCMLIR_DIR}/mlir/utils/performance/conv-configs
    ${TUNA_DIR}/tuna/rocmlir/import_configs.py --file_name ${ROCMLIR_DIR}/mlir/utils/jenkins/ci-configs/selected-gemm-configs --config_type gemm
                                                           #${ROCMLIR_DIR}/mlir/utils/performance/gemm-configs --config_type gemm
}


function tuning_run
{
    kind=$1
    baselabel=`date --iso-8601=minutes`

    ${TUNA_DIR}/tuna/go_fish.py rocmlir --init_session -l "$baselabel $kind" --config_type $kind 2> initlog
    session=`perl -n -e'/Added new session_id: (\d+)/ && print $1' < initlog`
    cat initlog
    ${TUNA_DIR}/tuna/rocmlir/load_job.py --session_id $session --config_type $kind
    (cd ${ROCMLIR_DIR}/build/ ; ${TUNA_DIR}/tuna/go_fish.py rocmlir --execute --session_id $session --config_type $kind)
    arch=`rocm_agent_enumerator  -name | awk -F: '{print $1;}'`
    ${TUNA_DIR}/tuna/rocmlir/export_configs.py  --session_id $session --config_type $kind -a -f mlir_tuning_${arch}.tsv
}

function tuna_run
{
    tuning_run convolution
    tuning_run gemm
}



export TUNA_DB_USER_NAME=root
export TUNA_DB_USER_PASSWORD=TunaTest
export TUNA_DB_HOSTNAME=localhost
export TUNA_DB_NAME=tuna
export PYTHONPATH=$TUNA_DIR:$PYTHONPATH

# First argument is rocMLIR directory;  if absent, assume we're in build.
if [ $# = 0 ]; then
    export ROCMLIR_DIR="$PWD/../.."
else
    export ROCMLIR_DIR=$1
fi


mysql_setup
tuna_setup
PYTHONPATH=$TUNA_DIR:$PYTHONPATH
tuna_populate
tuna_run
