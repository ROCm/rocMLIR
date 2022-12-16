#!/bin/bash

declare -A configArr
declare -a orders
declare -A configVarToCL
declare -A configCLToVar
## Define conv parameters in order
orders=(operation
        filterLayout
        inputLayout
        outputLayout
        groupSize
        batchSize
        inputChannel
        outputChannel
        inputHeight
        inputWidth
        filterHeight
        filterWidth
        dilationHeight
        dilationWidth
        strideHeight
        strideWidth
        paddingHeightLeft
        paddingHeightRight
        paddingWidthLeft
        paddingWidthRight
        tensorDataType
       )

## Command line options for each config parameter
## Convert variable to command line option
configVarToCL[operation]=operation
configVarToCL[filterLayout]=fil_layout
configVarToCL[inputLayout]=in_layout
configVarToCL[outputLayout]=out_layout
configVarToCL[groupSize]=groupsize # can also use g
configVarToCL[batchSize]=batchsize
configVarToCL[inputChannel]=in_channels
configVarToCL[outputChannel]=out_channels
configVarToCL[inputHeight]=in_h
configVarToCL[inputWidth]=in_w
configVarToCL[filterHeight]=fil_h
configVarToCL[filterWidth]=fil_w
configVarToCL[dilationHeight]=dilation_h
configVarToCL[dilationWidth]=dilation_w
configVarToCL[strideHeight]=conv_stride_h
configVarToCL[strideWidth]=conv_stride_w
## padding_h
configVarToCL[paddingHeightLeft]=padding_h_l
configVarToCL[paddingHeightRight]=padding_h_r
## padding_w
configVarToCL[paddingWidthLeft]=padding_w_l
configVarToCL[paddingWidthRight]=padding_w_r
configVarToCL[tensorDataType]=t

## Convert command line option to variable
configCLToVar[operation]=operation
configCLToVar[fil_layout]=filterLayout
configCLToVar[in_layout]=inputLayout
configCLToVar[out_layout]=outputLayout
configCLToVar[groupsize]=groupSize
configCLToVar[g]=groupSize
configCLToVar[batchsize]=batchSize
configCLToVar[in_channels]=inputChannel
configCLToVar[out_channels]=outputChannel
configCLToVar[in_h]=inputHeight
configCLToVar[in_w]=inputWidth
configCLToVar[fil_h]=filterHeight
configCLToVar[fil_w]=filterWidth
configCLToVar[dilation_h]=dilationHeight
configCLToVar[dilation_w]=dilationWidth
configCLToVar[conv_stride_h]=strideHeight
configCLToVar[conv_stride_w]=strideWidth
configCLToVar[padding_h_l]=paddingHeightLeft
configCLToVar[padding_h_r]=paddingHeightRight
configCLToVar[padding_w_l]=paddingWidthLeft
configCLToVar[padding_w_r]=paddingWidthRight
configCLToVar[t]=tensorDataType


usage() {
    cat <<END_USAGE
Usage: $PROG [options...]
       Search one or more conv configs in one or more conv config pools.

Options:
    -h|--help                  Display this help message
  Input Options:
    --input=                   Source of search. A test file.
                               Note: input files can be either mlir test files
                               (.mlir) or toml configuration files.
    --config                   Source of search. A single config.
  Target Options:
    --tomlfile=                One or more pools of conv configs as target of search.
                               If in search mode, multiple toml files can be selected
                               if comma separated, e.g.
                               $PROG --input=myTest --tomlfile=db1,db2 --search
                               If in insert mode, only one toml file is allowed.
                               Note:
                               1. if toml file is located at /mlir/test/e2e,
                                  then only the filename without extension is needed.
                                  Otherwiese, full file path is required.
                               2. If no toml files are provided, all toml files
                                  in /mlir/test/e2e will be used.
  Mode Options:
    -s|--search                Search mode, i.e. search the given configs, either
                               from input file(s) (--input=) or a single config
                               (--config=) from the pool(s).
                               This mode is set to be the default.
    -i|--insert                Insert mode, i.e. insert the given configs, either
                               from input file(s) (--input=) or a single config
                               (--config=) into the pool if not there yet.

Notes about config comparison:
1. Only geometric parameters (dim, strid, and padding) of the convolution problem
   is used. Directional (fwd, bwd, wrw) and storage related (data type, random input
   and arch) parameters are ignored.
2. Default values from rocmlir-gen.cpp are used for geometric parameters
   if they are not provided by user input

Example usages:

    $PROG --input=xxx.mlir -s # search tests in xxx.mlir in the default database
    $PROG --input=xxx.mlir --database=Resnet50Config -s # search tests in the Resnet50 databse
END_USAGE
}

function stderr() {
	cat - 1>&2
}

function print_convDB() {
    for db in "${convDB[@]}";
    do
        echo -n "    $db "
        if [ -f "${E2E_CONV_DB_DIR}/$db.toml" ];then
            printf "\xE2\x9C\x94\n"
        else
            printf "\xE2\x9D\x8C\n"
        fi
    done
}

##
## conv2d configs and their default values
##
function refresh_config_with_cmd_defaults() {
    configArr[operation]=conv2d
    configArr[num_cu]=64
    configArr[filterLayout]=kcyx
    configArr[inputLayout]=nchw
    configArr[outputLayout]=nkhw
    configArr[groupSize]=1
    configArr[batchSize]=-1
    configArr[inputChannel]=-1
    configArr[outputChannel]=-1
    configArr[inputHeight]=-1
    configArr[inputWidth]=-1
    configArr[filterHeight]=-1
    configArr[filterWidth]=-1
    configArr[dilationHeight]=1
    configArr[dilationWidth]=1
    configArr[strideHeight]=1
    configArr[strideWidth]=1
    configArr[paddingHeightLeft]=0
    configArr[paddingHeightRight]=0
    configArr[paddingWidthLeft]=0
    configArr[paddingWidthRight]=0
    configArr[tensorDataType]=f32
}

function print_config() {
    if [[ $populateDefaults -eq 0 ]]; then
        for config in "${orders[@]}";
        do
            if [[ "${FORMATS[*]}" =~ "no-dt" ]] && [[ "$config" == "tensorDataType" ]];then
                continue
            fi
            if [[ "${FORMATS[*]}" =~ "no-dir" ]] && [[ "$config" == "operation" ]];then
                continue
            fi
            if [[ "${FORMATS[*]}" =~ "no-layouts" ]] && [[ "$config" == *"Layout"* ]];then
                continue
            fi
            echo "-${configVarToCL[$config]}=${configArr[$config]}"
        done
    else
        print_config_with_defaults
    fi
}

function print_config_with_defaults() {
    if [[ ! "${FORMATS[*]}" =~ "no-dir" ]];then
        echo -n "-operation=${configArr[operation]} "
    fi
    if [[ ! "${FORMATS[*]}" =~ "no-layouts" ]];then
        echo -n "-fil_layout=${configArr[filterLayout]} "
        echo -n "-in_layout=${configArr[inputLayout]} "
        echo -n "-out_layout=${configArr[outputLayout]} "
    fi
    if [[ ! "${FORMATS[*]}" =~ "no-dt" ]];then
        echo -n "-t ${configArr[tensorDataType]} "
    fi
    echo "-p"
}


## process the input mlir test file or toml configuration file
##   1. Extract the config from each line
##   2. Extract values of each parameter and fill in configArr
##   3. Match the config from the database and insert if requested
function process_input() {
    ## Check input file format
    if [[ "${INPUT_OPTION}" =~ .mlir ]];then
        grep "rocmlir-gen" "${INPUT_OPTION}" > configs.tmp
    elif [[ "${INPUT_OPTION}" =~ .toml ]];then
        grep "config =" "${INPUT_OPTION}" > configs.tmp
    else
        echo -e "${COL_ERR}""ERROR: Unkonw format of input file: ${INPUT_OPTION}"`
                         `" (must be .mlir or .toml)" | stderr
        exit 0
    fi

    cnt=0
    while read -r line;
    do
        ((cnt++))
        echo "test $cnt in ${INPUT_OPTION##*/}"
        entry=$(canonicalize_config "$line")
        match_and_insert "$entry"
    done < configs.tmp
    rm configs.tmp
}

## Remove unuseful information from the given line or string
function preprocess_config() {
    ## $1: line or string that contains a config
    line=$1
    ## Only extract contents after rocmlir-gen
    line=${line%%|*}
    line=${line/\/\/\ RUN:\ rocmlir-gen/}
    #line=${line/--arch\ \%arch}
    ## remove lit defined variables
    line=${line/\%pv}
    line=${line/\%random_data}
    line=${line/\%rocmlir_gen_flags}
    ## remove spaces
    echo "$line" | xargs
}

## Format the given config
function canonicalize_config() {
    ## $1: target config
    ## return: formatted entry
    config=$(preprocess_config "$1")
    refresh_config_with_cmd_defaults
    update_config "$config"
    ## pretty print the config
    entry=$(print_config)
    echo "$entry" | xargs
}

## Use the given config ($1) to update the parameters in configArr
## and set populateDefaults if -p or -p=true is in the config
function update_config() {
    ## $1: config input
    str=$1
    ## replace -- with - for easy processing
    str=${str//--/-}
    populateDefaults=0
    while [[ "$str" == *"-"* ]];
    do
        ## Keep processing if there is an option in str
        ## extract the last key-value pair from str
        pair=$(expr "$str" : '.*\(-.*\)')
        ## replace = with space and remove - for easy processing
        pair=${pair//=/ }
        pair=${pair//-/}
        ## Remove the last key-value pair from str
        str=${str%-*}
        ## special case for -p
        if [[ "$pair" == "p" ]];then
            populateDefaults=1
        else
            ## key value pair
            key_value=($pair)
            key=${key_value[0]}
            value=${key_value[1]}
            ## special case: -p=true or -p true
            if [[ "$key" == "p" ]]; then
                if [[ $value == "true" ]];then
                    populateDefaults=1
                fi
            else
                ## Special case for layout: remove g
                if [[ "$key" == *"layout"* ]];then
                    value=${value/g/}
                fi
                ## special case for padding_
                if [[ "$key" == "padding_h" ]];then
                    configArr[paddingHeightLeft]=$value
                    configArr[paddingHeightRight]=$value
                elif [[ "$key" == "padding_w" ]];then
                    configArr[paddingWidthLeft]=$value
                    configArr[paddingWidthRight]=$value
                else
                    if [[ ${configCLToVar[$key]} ]];then
                        ## ignore cmd options not defined
                        ## E.g. -rand_type and -arch %arch/gfx1030
                        param=${configCLToVar[$key]}
                        configArr[$param]=$value
                    fi
                fi
            fi
        fi
    done
}

## Match the given config entry in the config_db
## and insert the entry into the db if not exist
function match_and_insert() {
    ## $1: config entry
    entry=$1
    entry=$(echo "$entry" | xargs)
    foundOne=0
    for db in "${DB_ARR[@]}";
    do
        if [ ! -f "$db" ];then
            db=${CONVDB_DIR}/$db.toml
        fi
        grep "config =" "${db}" > db.tmp
        dbN=1
        while read -r line;
        do
            entry_db=$(canonicalize_config "$line")
            if [[ "$entry" == "$entry_db" ]];then
                printf "    \xE2\x9C\x94 Match config %d in %s\n" $dbN "${db##*/}"
                foundOne=1
            fi
            ((dbN++))
        done < db.tmp
        ## If not found in the database, insert if MODE=insert
        if [[ $foundOne -eq 0 ]] && [[ "$MODE" == "insert" ]];then
            echo "    -> Insert new config into ${db##*/}"
            {
                echo ""
                echo "[[suite.test]]"
                echo "config = \"$entry\""
            } >> "$db"
        fi
        rm db.tmp
    done
    ## Not found in any database
    if [[ $foundOne -eq 0 ]] && [[ "$MODE" == "search" ]];then
        printf "    \xE2\x9D\x8C Not found in any database\n"
    fi
}


function check_database() {
    ## Set database to convDB if not set at command line
    DATABASE=${DATABASE_OPTION:-$convDB}
    IFS=',' read -r -a DB_ARR <<< "$DATABASE"
    err=0
    for db in "${DB_ARR[@]}";
    do
        if [ ! -f "$db" ] && [ ! -f "${CONVDB_DIR}/$db.toml" ];then
            err=1
            echo -e "${COL_ERR}""ERROR: $db does not exist" | stderr
        fi
    done
    [[ $err -eq 0 ]] || exit 0
    if [[ "$MODE" == "insert" ]] && [[ "$DATABASE" =~ "," ]];then
        echo -e "${COL_ERR}""ERROR: Only one database is allowed"`
                        ` "in insert mode" | stderr
        exit 1
    fi
}

function check_inputs() {
    if [[ "${INPUT_OPTION}" != "" ]]; then
        process_input
    elif [[ "${CONFIG_STR}" != "" ]];then
        entry=$(canonicalize_config "${CONFIG_STR}")
        match_and_insert "$entry"
    else
        echo -e "${COL_ERR}""ERROR: No input. Must provide input by" `
                         `"--input= or --config=" | stderr
        exit 1
    fi
}


function check_formats() {
    ## Overwrite the formats for insert mode as the default formats
    if [[ "$MODE" == "insert" ]] || [[ ${#FORMATS[@]} -eq 0 ]];then
        FORMATS=(no-dt no-dir no-layouts)
    fi
}


PROG=${0##*/}
PROG_DIR=$(cd "${0%/*}" && pwd -P)
ROCMLIR_DIR=$(cd "${PROG_DIR}" || return; git rev-parse --show-toplevel || echo "${HOME}/rocMLIR/")
CONVDB_DIR=${ROCMLIR_DIR}/mlir/test/e2e

convDB="${CONVDB_DIR}/MixedConvLayouts.toml"
convDB="$convDB,${CONVDB_DIR}/PaddedGemmConfig.toml"
convDB="$convDB,${CONVDB_DIR}/Resnet50Config.toml"
convDB="$convDB,${CONVDB_DIR}/PrResnet50.toml"
convDB="$convDB,${CONVDB_DIR}/Resnext101Config.toml"
convDB="$convDB,${CONVDB_DIR}/Resnet101Config.toml"
convDB="$convDB,${CONVDB_DIR}/conv2d_regression_fwd.toml"
convDB="$convDB,${CONVDB_DIR}/conv2d_regression_bwd.toml"
convDB="$convDB,${CONVDB_DIR}/conv2d_wrw_perf_config.toml"

populateDefaults=0

MODE=search
FORMATS=()

while (($#))
do
    case "$1" in
        -h | --help ) usage; exit 0 ;;
        --dt | --no-dt | --dir | --no-dir | --layouts | --no-layouts)
            FORMATS+=("${1#--}"); shift ;;
        -s | --search ) MODE=search; shift ;;
        -i | --insert ) MODE=insert; shift ;;
        --input=* | --database=* )
            option=${1%=*}
            option=${option#--}
            eval "${option^^}"_OPTION="${1#--*=}"
            shift
            ;;
        --config ) shift ;;
        *) CONFIG_STR+="$1 "; shift ;;
        esac
done

COL_ERR='\033[0:31m'
check_database
check_formats
check_inputs
