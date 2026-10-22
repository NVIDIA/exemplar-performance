# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for LLMB installer package.

This allows the package to be run as:
    python -m llmb_install
"""

if __name__ == "__main__":
    from llmb_install.cli import main

    main()
