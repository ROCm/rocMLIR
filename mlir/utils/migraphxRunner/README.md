## Installation

```bash
pip3 install -r ./requirements.txt
pip3 install -e .
```


## Actions

The runner gets configured through a `yaml` file (e.g., see `<src>/examples/config.yaml`). The configuration file is supposed to provide
paths to migraphx, rocMLIR, and a working directory. Moreover, the file contains a list of models. Each model provides the path
to its `onnx` file, data types and input paramters (if it is a dynamic model). All action listed below requires the user
to supply a config file.

### Collect

In this mode, the runner collecs `gemms` and `convs` for all provided models, data types, etc. using migraphx.

```bash
migraphRunner -c ./examples/config.yaml -a collect
```

The runner internally execute `migraphx-runner compile ...` for each test configuration.


## Join

The runner joins all `gemm` and `conv` files from different models into single ones, which can be supplied to `tuningRunner.py` (from rocmMLIR).
In this mode, the runner removes duplicated op, making the `tuningRunner.py` perform less work.

```bash
migraphRunner -c ./examples/config.yaml -a join
```

## Tune

The runner executes `tuningRunner.py` with all tuning configs collected in the previous step (see `Join`)

```bash
migraphRunner -c ./examples/config.yaml -a tune
```

## Adjust

`rocMLIR` and `MIGraphX` rely on different mechanisms to query the number of CUs. It can lead to some difference
on Navi cards. Therefore, the tuning dabase, obtained in the previous step (see `Tune`), may need to be adjusted.

`Adjust` action will make a backup copy of the tuning database, query the number of CUs using HIP API and make
an adjusted copy of it, which can be used in subsequent steps.

```bash
migraphRunner -c ./examples/config.yaml -a adjust
```


## Perf

The runner performs 3 runs for each model and stores outputs to dedicated files within the working directory, specified in the `yaml`-file.

The first run enables rocMLIR and supplies the tuning data base obtained in the previous step (see `Tune`). The second run enables only rocMLIR.
The tuning data base is not used during this run. The third run disables rocMLIR completely.

```bash
migraphRunner -c ./examples/config.yaml -a perf
```

## Report

The runner collects output data from the previous action (see `Perf`) and builds a markdown file for each model.

```bash
migraphRunner -c ./examples/config.yaml -a report
```
