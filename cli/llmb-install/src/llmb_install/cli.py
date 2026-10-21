# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for LLMB installer.

This module handles CLI argument parsing and provides the entry point for the installer.
"""

import argparse
import subprocess
import traceback

from llmb_install.constants import EXIT_CANCELLED


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    # Main parser
    parser = argparse.ArgumentParser(
        description="LLMB Workload Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmb-install                           # Interactive installation
  llmb-install -d                        # Interactive installation in dev mode
  llmb-install express /lustre/work      # Express mode with install path
  llmb-install -d express --workloads all   # Express mode in dev mode
  llmb-install --play config.yaml       # Fully automated from config file
        """,
    )

    # Global options (apply to all subcommands)
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help="Enable verbose output with debug logging",
    )

    parser.add_argument(
        '-i',
        '--image-folder',
        type=str,
        default=None,
        help="Path to a shared folder for container images. If provided, images will be stored here and symlinked into the installation directory.",
    )

    parser.add_argument(
        '-d',
        '--dev-mode',
        action='store_true',
        help="Development mode: Skip repository copying and use original repository location. Allows for version-controlled changes during development.",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Installation workflow', metavar='COMMAND')

    # Interactive mode (default) - no subcommand needed
    # Global options for when no subcommand is used
    interactive_group = parser.add_mutually_exclusive_group()

    interactive_group.add_argument(
        '--record',
        type=str,
        metavar='CONFIG_FILE',
        help="Record mode: Save user inputs to CONFIG_FILE without performing installation",
    )

    interactive_group.add_argument(
        '--play',
        type=str,
        metavar='CONFIG_FILE',
        help="Headless mode: Load configuration from CONFIG_FILE and install without prompts",
    )

    parser.add_argument(
        '--ui-mode',
        type=str,
        choices=['simple', 'rich'],
        default='simple',
        help="UI style for interactive mode: 'simple' for basic text, 'rich' for enhanced UI",
    )

    # Express subcommand
    express_parser = subparsers.add_parser(
        'express',
        help='Express installation using saved system configuration',
        description='Express mode performs minimal-prompt installation using previously saved system configuration. Requires a successful prior installation to create the saved config.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmb-install express /lustre/work                    # Prompt for workloads
  llmb-install express --install-path /lustre/work    # Prompt for workloads  
  llmb-install express --workloads all                # Prompt for install path
  llmb-install express /work --workloads pretrain_nemotron-h,pretrain_llama3.1  # Fully specified
  llmb-install -d express /work --workloads all       # Dev mode, no repo copy
        """,
    )

    express_parser.add_argument(
        'install_path_pos', nargs='?', help='Installation directory (can also use --install-path)'
    )

    express_parser.add_argument(
        '--install-path',
        type=str,
        dest='install_path_flag',
        help='Installation directory (alternative to positional argument)',
    )

    # Create mutually exclusive group for workload selection
    workload_group = express_parser.add_mutually_exclusive_group()

    workload_group.add_argument(
        '-w',
        '--workloads',
        type=str,
        help="Workloads to install: 'all' or comma-separated list (e.g. 'pretrain_nemotron-h,pretrain_llama3.1')",
    )

    workload_group.add_argument(
        '--exemplar',
        action='store_true',
        help="Install workloads specified in exemplar.yaml for this GPU type (Exemplar Cloud certification)",
    )

    express_parser.add_argument(
        '--list-workloads', action='store_true', help="List all available workloads for the saved GPU type and exit"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the LLMB installer."""
    from llmb_install.core.installer import Installer

    args = parse_arguments()
    installer = Installer()

    try:
        installer.run(args)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        raise SystemExit(EXIT_CANCELLED) from None
    except Exception as e:
        verbose = getattr(args, 'verbose', False)
        print(f"\nError: {type(e).__name__}: {e}")
        if not verbose:
            # The verbose traceback already includes the notes.
            for note in getattr(e, '__notes__', ()):
                print(note)
        if isinstance(e, subprocess.CalledProcessError):
            for output in (e.stdout, e.stderr):
                if output:
                    print(output.strip())
        if verbose:
            traceback.print_exc()
        else:
            print("Run with --verbose for the full traceback.")
        raise SystemExit(1) from e
