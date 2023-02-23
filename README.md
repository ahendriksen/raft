# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: Reusable Accelerated Functions and Tools</div>

## Investigation of compile times

The goal of this branch is to investigate how compilation times are impacted by
various design decisions. We have picked a single translation unit
(`l1_float_float_float_int.cu`) and compiled for a single architecture (SM 70).
Compile times are tracked over a series of commits:

| Commit                                                            | Compilation time | Message                               |
|:------------------------------------------------------------------|------------------|---------------------------------------|
| [dfc1274e1](https://github.com/ahendriksen/raft/commit/dfc1274e1) | 20.2s            | Add baseline compile time             |
| [069df315a](https://github.com/ahendriksen/raft/commit/069df315a) | 17.8s            | Remove cutlass includes               |
| [09a39e7f2](https://github.com/ahendriksen/raft/commit/09a39e7f2) | 12.7s            | Only compile the kernel dispatch code |
| [631ecd8ac](https://github.com/ahendriksen/raft/commit/631ecd8ac) | 11.9s            | Minimize header files                 |
| [8ff45bc00](https://github.com/ahendriksen/raft/commit/8ff45bc00) | 5.7s             | Do not include rmm                    |
| [ce07d6d1e](https://github.com/ahendriksen/raft/commit/ce07d6d1e) | 10.4s            | Add back spdlog                       |
| [90c8e07c6](https://github.com/ahendriksen/raft/commit/90c8e07c6) | 11.2s            | Add back rmm                          |
| [c47299975](https://github.com/ahendriksen/raft/commit/c47299975) | 4.8s             | Remove string and exceptions          |
| [cf47e5df2](https://github.com/ahendriksen/raft/commit/cf47e5df2) | 2.8s             | pragma unroll 1                       |

Using `nvcc`'s `--time` option, we can get an overview of where the time is spent during compilation:

| version                 | cicc | cudafe++ | fatbinary | gcc (compiling) | gcc (preprocessing 1) | gcc (preprocessing 4) | nvcc (driver) | ptxas | total |
|:------------------------|-----:|---------:|----------:|----------------:|----------------------:|----------------------:|--------------:|------:|------:|
| baseline                |  7.7 |      3.1 |       0.0 |             6.7 |                   0.7 |                   0.7 |           0.1 |   0.8 |  19.7 |
| 01_no_cutlass           |  7.1 |      2.5 |       0.0 |             6.2 |                   0.5 |                   0.5 |           0.0 |   0.8 |  17.7 |
| 02_only_kernel_dispatch |  5.9 |      1.7 |       0.0 |             3.5 |                   0.3 |                   0.3 |           0.0 |   0.8 |  12.6 |
| 04_remove_rmm           |  2.7 |      0.6 |       0.0 |             1.1 |                   0.2 |                   0.2 |           0.0 |   0.8 |   5.6 |
| 05_add_back_spdlog      |  5.1 |      1.1 |       0.0 |             2.9 |                   0.2 |                   0.2 |           0.0 |   0.8 |  10.4 |
| 06_add_back_spdlog_rmm  |  5.5 |      1.2 |       0.0 |             3.1 |                   0.3 |                   0.3 |           0.0 |   0.8 |  11.1 |
| 07_no_string_exception  |  2.5 |      0.4 |       0.0 |             0.6 |                   0.2 |                   0.2 |           0.0 |   0.8 |   4.8 |
| 08_pragma_unroll_1      |  1.0 |      4.0 |       0.0 |             0.6 |                   0.2 |                   0.2 |           0.0 |   0.2 |   2.7 |


## Conclusions

### Obtain 30% compile time reduction by limiting compilation of non-device code
- [069df315a](https://github.com/ahendriksen/raft/commit/069df315a) 17.8s : Remove cutlass includes
- [09a39e7f2](https://github.com/ahendriksen/raft/commit/09a39e7f2) 12.7s : Only compile the kernel dispatch code

Compiling only the kernel launch dispatch code takes 12.7s, whereas compiling
the middle layer between public api and dispatch takes an additional 5s for a
total of 17.8s.

### Obtain 50% compile time reduction by not including rmm and spdlog
- [631ecd8ac](https://github.com/ahendriksen/raft/commit/631ecd8ac) 11.9s : Minimize header files
- [8ff45bc00](https://github.com/ahendriksen/raft/commit/8ff45bc00)  5.7s : Do not include rmm
- [ce07d6d1e](https://github.com/ahendriksen/raft/commit/ce07d6d1e) 10.4s : Add back spdlog
- [90c8e07c6](https://github.com/ahendriksen/raft/commit/90c8e07c6) 11.2s : Add back rmm

By just removing the include of `rmm` in commit
[8ff45bc00](https://github.com/ahendriksen/raft/commit/8ff45bc00), the compile
times were reduced from 11.9s to 5.7s. This is a big improvement! The reason
that compilation takes much longer is that `spdlog` instantiates a bunch of
templates in every translation unit when used as a header only library. This
happens in
[pattern_formatter::handle_flag_](https://github.com/gabime/spdlog/blob/da14258533cb951ce85087ceb45556e0b8253660/include/spdlog/pattern_formatter-inl.h#L1105),
which is instantiated
[here](https://github.com/gabime/spdlog/blob/da14258533cb951ce85087ceb45556e0b8253660/include/spdlog/pattern_formatter-inl.h#L1410).
Just adding back the `spdlog` header doubles the compile times of `cicc` (device
side) and also `gcc` on the host side.

### Obtain a 15% compile time reduction by not using/including strings and exceptions
- [8ff45bc00](https://github.com/ahendriksen/raft/commit/8ff45bc00)  5.7s : Do not include rmm
- [c47299975](https://github.com/ahendriksen/raft/commit/c47299975)  4.8s : Remove string and exceptions

We can reduce compile times from 5.7s to 4.8s by not using exceptions and strings. 

### Compile times can be further reduced by using pragma unroll 1
- [c47299975](https://github.com/ahendriksen/raft/commit/c47299975)  4.8s : Remove string and exceptions
- [cf47e5df2](https://github.com/ahendriksen/raft/commit/cf47e5df2)  2.8s : pragma unroll 1

Compile times can be reduced from 4.8s to 2.8s by using pragma unroll 1. The
impact on cicc (2.5s -> 1.0s) and ptxas (0.8 -> 0.2s) is quite large. The
runtime performance of the kernel will suffer as a result. It might be wort
judiciously tweaking the pragma unrolls.

## Methodology

The project was configured with: 

``` sh
cmake -S cpp -B cpp/build \
   -DCMAKE_INSTALL_PREFIX= \
   -DCMAKE_CUDA_ARCHITECTURES=70 \
   -DCMAKE_BUILD_TYPE=Release \
   -DRAFT_COMPILE_LIBRARIES=OFF \
   -DRAFT_ENABLE_NN_DEPENDENCIES=OFF \
   -DRAFT_NVTX=ON \
   -DDISABLE_DEPRECATION_WARNINGS=ON \
   -DBUILD_TESTS=ON \
   -DBUILD_BENCH=ON \
   -DCMAKE_MESSAGE_LOG_LEVEL=  \
   -DRAFT_COMPILE_NN_LIBRARY=OFF \
   -DRAFT_COMPILE_DIST_LIBRARY=ON \
   -DRAFT_USE_FAISS_STATIC=OFF \
   -DRAFT_ENABLE_thrust_DEPENDENCY=ON \
   -DFIND_RAFT_CPP=ON \
   -DCUDA_LOG_COMPILE_TIMES=ON \
   -G Ninja
```

A single translation unit was picked (`l1_float_float_float_int.cu`) for
benchmarking. The compilation command was obtained with

``` sh
ninja -C cpp/build -t commands CMakeFiles/raft_distance_lib.dir/src/distance/distance/specializations/detail/l1_float_float_float_int.cu.o
```

This resulted in the following build command, where the `--time` flag was added
to save the intermediate time steps to a csv file.

``` sh
cd cpp/build
time nvcc \
   --time=../../nvcc_compile_08_pragma_unroll_1.csv \
   -forward-unknown-to-host-compiler -DCUTLASS_NAMESPACE=raft_cutlass \
   -DFMT_HEADER_ONLY=1 -DNVTX_ENABLED -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DSPDLOG_FMT_EXTERNAL \
   -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
   -Draft_distance_lib_EXPORTS \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/rmm-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/thrust-src \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/thrust-src/dependencies/cub \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/fmt-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/spdlog-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/cuco-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/libcudacxx-src/lib/cmake/libcudacxx/../../../include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/nvidiacutlass-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/nvidiacutlass-build/include \
   -isystem=/nix/store/sivnyagc6j0skwfgwhy2a4hlwy972kic-mycudatoolkit/include \
   -O3 -DNDEBUG --generate-code=arch=compute_70,code=[compute_70,sm_70] \
   -Xcompiler=-fPIC -Xcompiler=-Wno-deprecated-declarations \
   --expt-extended-lambda --expt-relaxed-constexpr \
   -DCUDA_API_PER_THREAD_DEFAULT_STREAM \
   -Xfatbin=-compress-all  -Werror=all-warnings -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations \
   -Xcompiler=-fopenmp -std=c++17 \
   -MD -MT CMakeFiles/raft_distance_lib.dir/src/distance/distance/specializations/detail/l1_float_float_float_int.cu.o \
   -MF CMakeFiles/raft_distance_lib.dir/src/distance/distance/specializations/detail/l1_float_float_float_int.cu.o.d \
   -x cu \
   -c /home/ahendriksen/projects/raft-spdlog-issue/cpp/src/distance/distance/specializations/detail/l1_float_float_float_int.cu \
   -o CMakeFiles/raft_distance_lib.dir/src/distance/distance/specializations/detail/l1_float_float_float_int.cu.o
```
This command was repeated in each commit and timings are reported.
