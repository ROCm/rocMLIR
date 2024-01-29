#!/bin/bash

set +x

export TUNA_PROXYCMD=

function send_files_to_database
{
    arch=$1
    dname="csv-$arch-$(date --iso-8601=minutes)"

    # csv.cnf
    echo -e "[client]\nuser=${TUNA_REMOTE_DB_USER_NAME}\npassword=${TUNA_REMOTE_DB_USER_PASSWORD}\ndatabase=${TUNA_REMOTE_DB_NAME}\nlocal-infile=1\nhost=127.0.0.1" > csv.cnf

    # Compensate for bad name choices.
    if [ -e "convolution-results.csv" ]; then
        mv convolution-results.csv conv-results.csv
        mv convolution-config.csv conv-config.csv
        mv convolution-session.csv conv-session.csv
    fi

    # csv-commands.sql
    # Lock the tables because we could have another node (with a different chip)
    # trying to upload at the same time, and we count on knowing which session
    # was the most recently loaded.
    echo -n "lock tables session_rocmlir write" > csv-commands.sql
    for kind in conv gemm attention
    do if [ -e "${kind}-results.csv" ]
       then echo -n ", rocmlir_${kind}_results write, rocmlir_${kind}_config write" >> csv-commands.sql
       fi
    done
    echo ";" >> csv-commands.sql
    # Load session first to acquire new session ID (via autoincrement when ID
    # field is ignored), then configs because results has a foreign-key
    # dependence on config.id, then results using max(id) to get new session ID.
    for kind in conv gemm attention
    do if [ -e "${kind}-results.csv" ]
       then echo "load data local infile '${dname}/${kind}-session.csv' into table session_rocmlir fields terminated by ',' (@dummy, insert_ts, update_ts, valid, arch, num_cu, rocm_v, @dummy, @dummy, @dummy, mlir_v, arch_full, config_type, tuning_space);" >> csv-commands.sql
            echo "load data local infile '${dname}/${kind}-config.csv' into table rocmlir_${kind}_config fields terminated by ',';" >> csv-commands.sql
            echo "load data local infile '${dname}/${kind}-results.csv' into table rocmlir_${kind}_results fields terminated by ',' enclosed by '\"' (@dummy, insert_ts, update_ts, valid, config_str, perf_config, kernel_tflops, config, @session) set session = (select max(id) from session_rocmlir);" >> csv-commands.sql
            echo 'show warnings;' >> csv-commands.sql
       fi
    done
    echo 'unlock tables;' >> csv-commands.sql

    common_opts='-i ./keyfile -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null'
    if [ -n "${TUNA_PROXYCMD}" ]; then
        common_opts+=" -oProxyCommand=\"${TUNA_PROXYCMD}\""
    fi
    url_base="//${TUNA_REMOTE_DB_USER_NAME}@${TUNA_REMOTE_DB_HOSTNAME}:${TUNA_REMOTE_DB_PORT}"

    # send files to database server
    echo sftp ${common_opts} sftp:${url_base}
    sftp ${common_opts} sftp:${url_base} <<foo
      mkdir ${dname}
      cd ${dname}
      put *-results.csv
      put *-config.csv
      put *-session.csv
      put csv.cnf
      put csv-commands.sql
foo

    # do database things
    echo ssh ${common_opts} ssh:${url_base} "mysql --defaults-extra-file=${dname}/csv.cnf --verbose < ${dname}/csv-commands.sql"
    ssh ${common_opts} ssh:${url_base} "mysql --defaults-extra-file=${dname}/csv.cnf --verbose < ${dname}/csv-commands.sql"

    # clean up, when I'm confident it's working.
    #ssh ${common_opts} ssh:${url_base} "rm -rf ${dname}"
}

ARCH="$1"
[ -n "$ARCH" ] && send_files_to_database "$ARCH"
