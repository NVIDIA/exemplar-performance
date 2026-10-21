# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NSys download source routing (NSYS_SOURCES) and download_file TLS handling."""

import ssl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmb_install.constants import NSYS_SOURCES, NsysSource
from llmb_install.downloads.tools import _install_nsys, _nsys_download_spec
from llmb_install.utils.download import download_file

PUBLIC_VERSION = '2025.5.1.121-3638078'
NSYS_PUBLIC_BASE_URL = 'https://developer.download.nvidia.com/devtools/nsight-systems'

# An extra source listed before the public fallback, with TLS verification off.
CUSTOM_VERSION = '2026.1.0.1-custom'
CUSTOM_BASE_URL = 'https://nsys.example.invalid/builds'
CUSTOM_SOURCE = NsysSource(
    version_pattern=r'-custom$',
    base_url=CUSTOM_BASE_URL,
    url_template='{base_url}/{version}/{filename}',
    cache_name_template='{version}-{filename}',
    filenames={'x86_64': 'nsys.run', 'aarch64': 'nsys-sbsa.run'},
    verify_tls=False,
)
WITH_CUSTOM_SOURCE = patch('llmb_install.downloads.tools.NSYS_SOURCES', [CUSTOM_SOURCE, *NSYS_SOURCES])


class TestNsysDownloadSpec(unittest.TestCase):
    """Routing of pinned NSys versions to download sources."""

    def test_public_version_routes_to_public_server(self):
        for arch, filename in (
            ('x86_64', f'NsightSystems-linux-public-{PUBLIC_VERSION}.run'),
            ('aarch64', f'NsightSystems-linux-sbsa-public-{PUBLIC_VERSION}.run'),
        ):
            with self.subTest(arch=arch):
                spec = _nsys_download_spec(PUBLIC_VERSION, arch)
                self.assertEqual(spec.url, f'{NSYS_PUBLIC_BASE_URL}/{filename}')
                self.assertEqual(spec.cache_name, filename)
                self.assertTrue(spec.verify_tls)

    @WITH_CUSTOM_SOURCE
    def test_first_matching_source_wins(self):
        spec = _nsys_download_spec(CUSTOM_VERSION, 'aarch64')
        self.assertEqual(spec.url, f'{CUSTOM_BASE_URL}/{CUSTOM_VERSION}/nsys-sbsa.run')
        self.assertEqual(spec.cache_name, f'{CUSTOM_VERSION}-nsys-sbsa.run')
        self.assertFalse(spec.verify_tls)
        # Versions the custom pattern does not match still fall through to the public server.
        self.assertTrue(_nsys_download_spec(PUBLIC_VERSION, 'x86_64').url.startswith(NSYS_PUBLIC_BASE_URL))

    def test_unsupported_arch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _nsys_download_spec(PUBLIC_VERSION, 'ppc64le')
        self.assertIn('Unsupported architecture', str(ctx.exception))

    def test_no_matching_source_raises(self):
        """A version no source matches must raise, not return None and fail later at the caller."""
        with patch('llmb_install.downloads.tools.NSYS_SOURCES', []):
            with self.assertRaises(ValueError) as ctx:
                _nsys_download_spec(PUBLIC_VERSION, 'x86_64')
        self.assertIn('No NSys download source matches', str(ctx.exception))


class TestInstallNsysVerification(unittest.TestCase):
    """_install_nsys must pass the source's TLS policy to download_file and warn when disabled.

    Runs against a real temporary tree (no stubbed os.path.exists): the cached
    installer is simply absent, the installed nsys binary is pre-created, and
    the mocked download_file writes a real file so the cache rename actually happens.
    """

    def _run_install_nsys(self, version, arch):
        captured = {}

        def fake_download(url, dest_path, *, verify=True):
            Path(dest_path).write_bytes(b'fake installer')
            captured['url'] = url
            captured['verify'] = verify

        with tempfile.TemporaryDirectory() as tmp:
            install_path = Path(tmp)
            install_dir = install_path / 'nsys' / version
            (install_dir / 'bin').mkdir(parents=True)
            (install_dir / 'bin' / 'nsys').write_text('#!/bin/sh\n')

            with (
                patch('llmb_install.downloads.tools.download_file', side_effect=fake_download),
                patch('llmb_install.downloads.tools.logger') as mock_logger,
                patch('llmb_install.downloads.tools.subprocess.run'),
            ):
                _install_nsys(version, str(install_dir), str(install_path), arch)
                cache_dir = install_path / '.cache' / 'tools' / 'nsys'
                # Collect inside the TemporaryDirectory; the paths die with it.
                cached = [(p.name, p.read_bytes()) for p in sorted(cache_dir.iterdir())]

        return captured, mock_logger, cached

    @WITH_CUSTOM_SOURCE
    def test_unverified_source_disables_tls_and_warns(self):
        captured, mock_logger, cached = self._run_install_nsys(CUSTOM_VERSION, 'aarch64')
        self.assertEqual(captured['url'], f'{CUSTOM_BASE_URL}/{CUSTOM_VERSION}/nsys-sbsa.run')
        self.assertFalse(captured['verify'])
        mock_logger.warning.assert_called_once()
        self.assertIn('TLS certificate verification', mock_logger.warning.call_args[0][0])
        self.assertEqual(cached, [(f'{CUSTOM_VERSION}-nsys-sbsa.run', b'fake installer')])

    def test_public_build_keeps_tls(self):
        captured, mock_logger, cached = self._run_install_nsys(PUBLIC_VERSION, 'x86_64')
        self.assertTrue(captured['verify'])
        mock_logger.warning.assert_not_called()
        self.assertEqual(cached, [(f'NsightSystems-linux-public-{PUBLIC_VERSION}.run', b'fake installer')])


class _FakeResponse:
    """Minimal urlopen response returning fixed chunks."""

    def __init__(self, chunks, headers=None):
        self._chunks = list(chunks)
        self.headers = headers or {}

    def read(self, size=-1):
        return self._chunks.pop(0) if self._chunks else b''

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestDownloadFileVerifyFlag(unittest.TestCase):
    """download_file builds its SSL context from the verify flag."""

    def _download_with_context(self, verify):
        with patch('llmb_install.utils.download.urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse([b'hello'], {'Content-Length': '5'})
            with tempfile.TemporaryDirectory() as tmp:
                download_file('https://example.invalid/file', Path(tmp) / 'file', verify=verify)
        context = mock_urlopen.call_args.kwargs['context']
        timeout = mock_urlopen.call_args.kwargs['timeout']
        return context, timeout

    def test_verify_true_uses_verified_context(self):
        context, timeout = self._download_with_context(True)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertGreater(timeout, 0)

    def test_verify_false_uses_unverified_context(self):
        context, _ = self._download_with_context(False)
        self.assertFalse(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_NONE)

    def test_written_content_matches_response(self):
        with patch('llmb_install.utils.download.urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse([b'hello ', b'world'], {'Content-Length': '11'})
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / 'file'
                download_file('https://example.invalid/file', dest)
                self.assertEqual(dest.read_bytes(), b'hello world')

    def test_truncated_download_raises(self):
        """A connection cut before Content-Length is reached must not look like success."""
        with patch('llmb_install.utils.download.urllib.request.urlopen') as mock_urlopen:
            # Header claims 10 bytes; the server delivers only 3.
            mock_urlopen.return_value = _FakeResponse([b'abc'], {'Content-Length': '10'})
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / 'file'
                with self.assertRaises(RuntimeError) as ctx:
                    download_file('https://example.invalid/file', dest)
                self.assertIn('incomplete download', str(ctx.exception))
                self.assertIn('received 3 of 10 bytes', str(ctx.exception))

    def test_missing_content_length_is_accepted(self):
        """Responses without Content-Length (e.g. chunked) cannot be length-checked."""
        with patch('llmb_install.utils.download.urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse([b'abc'])
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / 'file'
                download_file('https://example.invalid/file', dest)
                self.assertEqual(dest.read_bytes(), b'abc')


if __name__ == '__main__':
    unittest.main()
