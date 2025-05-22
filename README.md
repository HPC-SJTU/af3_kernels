# AlphaFold3 Kernels

## Introduction

This repository contains the kernels for AlphaFold3, which are optimized for performance and efficiency. The kernels are designed to work with PyTorch and can be used to accelerate inference of AlphaFold3 models on CPU.

## Pre-requisite

gcc v8.3.0 or higher (suggested version: 13.3.0)

PyTorch v2.0.0 or higher (suggested version: 2.6.0)

## Installation

```bash
$ git submodule update --init
$ make install
```

## License

This project is derived from [tpp-pytorch-extension](https://github.com/libxsmm/tpp-pytorch-extension.git), originally licensed under the BSD 3-Clause License. The original license is included in the [LICENSE](LICENSE) file.

Copyright (c) 2025, Xflops. All rights reserved.
