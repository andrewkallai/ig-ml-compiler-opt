# input-gen utilities

## Dependencies

input-gen requires a C++20 conformant standard library.

### Python configuration
Note that this repo requires python `3.11`.

Example compilation configuration for python:
``` shell
./configure --prefix=/path/to/install/dir --enable-shared --enable-loadable-sqlite-extensions --enable-optimizations
make -j
make install
```

### LLVM build configuration

Baseline cmake configuration for the llvm installation:

<<<<<<< HEAD
``` 
cmake $LLVM_PROJECT_ROOT/llvm -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="compiler-rt" -DCOMPILER_RT_BUILD_INPUTGEN="ON" -DCMAKE_BUILD_TYPE=Release
```
These options are required: `-DLLVM_ENABLE_RUNTIMES="compiler-rt" -DCOMPILER_RT_BUILD_INPUTGEN="ON"`
=======
*   Recent Ubuntu distro, e.g. 22.04
*   python 3.10.x/3.11.x
*   for local training, which is currently the only supported mode, we recommend
    a high-performance workstation (e.g. 96 hardware threads).
>>>>>>> 4b3511540acfc111ffbc23c254d90786eb29c86d

See the following for more details on building LLVM:  https://llvm.org/docs/CMake.html

## Scripts

The input-gen scripts generally accept the option `-mclang` to specify
additional flags to `clang` and `-mllvm` for additional options to `opt`.

Use `--debug` to enable verbose output for debugging purposes.

Use `--one <id>` to process a specific row id from the input dataset.

`clang` is used for linking and compiling so depending on your environment,
additional flags may need to be specified.

For example to use a newer gcc toolchain rather than the default (required when
your default C++ std lib may be too old)
``` shell
-mclang='--gcc-toolchain=/path/to/gcc-toolchain' 
```

<<<<<<< HEAD
Multiple flags can also be specified as such:
``` shell
-mclang='--flag1' -mclang='--flag2' 
=======
The actual dependencies:
```shell
./versioned_pipenv sync --system --categories "packages dev-packages ci"
```
Note that the above command will only work from the root of the repository
since it needs to have `Pipfile.lock` in the working directory at the time
of execution.

The above command will also install all the packages, including development
packages (the `dev-packages` category), and packages only needed in CI (the
`ci` category). If you do not need those, you can omit them from the categories
option.

Optionally, to run tests (run_tests.sh), you also need:

```shell
sudo apt-get install virtualenv
>>>>>>> 4b3511540acfc111ffbc23c254d90786eb29c86d
```

### Generating inputs for a module

``` shell
python3 input_gen.py --module input_module.ll (--entry-function=ENTRY_FUNCTION | --entry-all | --entry-marked)
```

### Generating ComPileLoop from ComPile

``` shell
PYTHONPATH=$PYTHONPATH:. python3 -m com_pile_utils.generate_com_pile_loop --dataset ~/datasets/ComPile/ --output-dataset ~/datasets/ComPileLoop.sqlite3
```

USR1 can be sent to get a status report, USR2 can be sent to abort and write out
the current pending database file.

### Generating ComPileLoop+Inputs from ComPileLoop

``` shell
PYTHONPATH=$PYTHONPATH:. python3 -m com_pile_utils.generate_com_pile_loop_inputs --dataset ~/datasets/ComPileLoop.sqlite3 --output-dataset ~/datasets/ComPileLoopInputs.sqlite3
```

### Demo of how to process ComPileLoop

``` shell
python3 process_com_pile_loop_demo.py --dataset ~/datasets/ComPileLoop/
```

### Demo of how to process ComPileLoop+Inputs

``` shell
python3 process_com_pile_loop_inputs_demo.py --dataset ~/datasets/ComPileLoopInputs/
```

This will read the dataset and replay all inputs in it.

### Generating samples for training the unroll heuristic

Install the dependencies:

``` shell
dnf install libpfm-devel
```

From the root of the repo, change to the profiling runtime directory.

``` shell
cd compiler_opt/sl/unrolling/rts/
make CPU=AMD # For AMD CPUs
make CPU=INTEL # For Intel CPUs
```

Check the compiled timing runtime:

```
make check
```

And go back to the root.

```
cd -
```

This should result in the following file:

``` shell
compiler_opt/sl/unrolling/rts/unrolling_profiler.o
```

The following can be used to generate training samples
``` shell
./compiler_opt/sl/unrolling/ray_start_single_node_generate_unroll_training_samples.sh
```
The script may need to be edited depending on your CPU configuration (number of physical and logical cores).

The training samples must be preprocessed using the following before training:

``` shell
PYTHONPATH=$PYTHONPATH:. python3 -m compiler_opt.sl.unrolling.preprocess_unroll_training_samples --dataset ~/datasets/UnrollTrainingSamples.sqlite3 --output-dataset ~/datasets/UnrollTrainingSamplesPreprocessed.sqlite3 --output-parquet ~/datasets/UnrollTrainingSamplesPreprocessed.parquet
```

### Training the unroll heuristic

``` shell
PYTHONPATH=$PYTHONPATH:. python3 -m compiler_opt.sl.unrolling.train --preprocessed-dataset ~/datasets/UnrollTrainingSamplesPreprocessed.parquet
```
