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