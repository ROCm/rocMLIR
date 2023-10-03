#!/bin/bash

export TUNA_DB_USER_NAME=rocmlir_user
export TUNA_DB_USER_PASSWORD="mlir<234"
export TUNA_DB_HOSTNAME=127.0.0.1
export TUNA_DB_NAME=rocmlir


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

function remote_mysql_setup
{
    # Make ssh tunnel to ixt-rack-15.  Assumes an ssh agent.
    rack15addr=10.216.64.100
    rack15port=20057
    commonopts="-4 -f -N -p $rack15port -L3306:localhost:3306 -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -l $TUNA_DB_USER_NAME"

    # First case is simple hosts, second is lockhart, third is other 10.216.64.100 hosts.
    status=`ssh $commonopts $rack15addr ||
            ssh $commonopts -oProxyCommand='nc -X connect -x 172.23.0.23:3128 %h %p' $rack15addr ||
            ssh $commonopts -oProxyCommand='ssh -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -W %h:%p fpadmin@rocmhead.amd.com' $rack15addr`

    return $status
}


function tuna_setup
{
    startdir=`pwd`
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

    export PYTHONPATH=$TUNA_DIR:$PYTHONPATH

    cd $startdir

    tables=`python3 <<foo
import pymysql
from sqlalchemy import create_engine, inspect
e = create_engine(f"mysql+pymysql://${TUNA_DB_USER_NAME}:${TUNA_DB_USER_PASSWORD}@${TUNA_DB_HOSTNAME}:3306/${TUNA_DB_NAME}")
i = inspect(e)
print(i.get_table_names())
foo
`
#     tables=`mysql -h $TUNA_DB_HOSTNAME -u $TUNA_DB_USER_NAME -p${TUNA_DB_USER_PASSWORD} --database $TUNA_DB_NAME \
#                   -e 'show tables;' 2>/dev/null`

    if [ $? -ne 0 ] ; then
        echo Database table probe failed.
        exit 2
    fi
    if [ "$tables" = "[]" ]; then
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
    mysql -h $TUNA_DB_HOSTNAME --database $TUNA_DB_NAME -u $TUNA_DB_USER_NAME -p$TUNA_DB_USER_PASSWORD \
          -e "delete from rocmlir_${tablekind}_results;"
    mysql -h $TUNA_DB_HOSTNAME --database $TUNA_DB_NAME -u $TUNA_DB_USER_NAME -p$TUNA_DB_USER_PASSWORD \
          -e "delete from rocmlir_${tablekind}_job;"
    mysql -h $TUNA_DB_HOSTNAME --database $TUNA_DB_NAME -u $TUNA_DB_USER_NAME -p$TUNA_DB_USER_PASSWORD \
          -e "delete from rocmlir_${tablekind}_config;"
}

function tuna_run
{
    kind=$1
    baselabel=`date --iso-8601=minutes`

    clear_tables $kind
    ${TUNA_DIR}/tuna/rocmlir/import_configs.py --file_name ${CONFIGS_FILE} --config_type $kind
    ${TUNA_DIR}/tuna/go_fish.py rocmlir --init_session -l "$baselabel $kind" --config_type $kind 2> initlog
    session=`perl -n -e'/Added new session_id: (\d+)/ && print $1' < initlog`
    cat initlog
    ${TUNA_DIR}/tuna/rocmlir/load_job.py --session_id $session --config_type $kind
    (cd ${ROCMLIR_DIR}/build/ ; ${TUNA_DIR}/tuna/go_fish.py rocmlir --execute --session_id $session --config_type $kind)
    ${TUNA_DIR}/tuna/rocmlir/export_configs.py --session_id $session --config_type $kind --append -f "$OUT_FILE"
}



usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage

export CONFIGS_FILE=
export TUNA_DIR=`pwd`/MITuna      # Assumes we're in the build directory
export ROCMLIR_DIR=`pwd`/..       # Assumes we're in the build directory
export OUT_FILE=results.tsv
export OP=convolution

# -c configs
# -t tunadir
# -r rocmlirdir
# -f outfile
# -o operation
while getopts ":hc:t:r:f:o:s" arg; do
  case $arg in
    o) # Operation (convolution or gemm [default convolution])
      OP=${OPTARG}
      [ "$OP" = "convolution" -o "$OP" = "gemm" ] \
        || echo "Operation needs to be either 'convolution' or 'gemm'."
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
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done


set -x

export PYTHONPATH=$TUNA_DIR:$PYTHONPATH


# if ! pgrep mysqld ; then
#     mysqld -D
#     tuna_setup
# fi

if [ "$VIRTUAL_ENV" = "" ]; then
    source ${TUNA_DIR}/myvenv/bin/activate
fi

if ! remote_mysql_setup ; then
    echo Failed to connect to database.
    exit 1
fi
tuna_setup
tuna_run $OP
