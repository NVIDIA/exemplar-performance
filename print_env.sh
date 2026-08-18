#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Reports relevant environment information useful for diagnosing and
# debugging Exemplar Performance issues.
# Usage:
# "./print_env.sh" - prints to stdout
# "./print_env.sh > env.txt" - prints to file "env.txt"

print_env() {
    echo "***git***"
    if [ "$(git rev-parse --is-inside-work-tree 2> /dev/null)" == "true" ]; then
        git log --decorate -n 1
        echo "***git submodules***"
        git submodule status --recursive
    else
        echo "Not inside a git repository"
    fi
    echo

    echo "***OS Information***"
    cat /etc/*-release
    uname -a
    echo

    echo "***GPU Information***"
    nvidia-smi
    echo

    echo "***CPU***"
    lscpu
    echo

    echo "***CMake***"
    command -v cmake && cmake --version
    echo

    echo "***g++***"
    command -v g++ && g++ --version
    echo

    echo "***nvcc***"
    command -v nvcc && nvcc --version
    echo

    echo "***Python***"
    command -v python && python -c "import sys; print('Python {0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))"
    echo

    echo "***Environment Variables***"

    printf '%-32s: %s\n' PATH "$PATH"

    printf '%-32s: %s\n' LD_LIBRARY_PATH "$LD_LIBRARY_PATH"

    printf '%-32s: %s\n' NUMBAPRO_NVVM "$NUMBAPRO_NVVM"

    printf '%-32s: %s\n' NUMBAPRO_LIBDEVICE "$NUMBAPRO_LIBDEVICE"

    printf '%-32s: %s\n' CONDA_PREFIX "$CONDA_PREFIX"

    printf '%-32s: %s\n' PYTHON_PATH "$PYTHON_PATH"

    echo

    # Print conda packages if conda exists
    if type "conda" &> /dev/null; then
        echo '***conda packages***'
        command -v conda && conda list
        echo
    # Print pip packages if pip exists
    elif type "pip" &> /dev/null; then
        echo "conda not found"
        echo "***pip packages***"
        command -v pip && pip list
        echo
    else
        echo "conda not found"
        echo "pip not found"
    fi
}

echo "<details><summary>Click here to see environment details</summary><pre>"
echo "     "
print_env | while read -r line; do
    echo "     $line"
done
echo "</pre></details>"
