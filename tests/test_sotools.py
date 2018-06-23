#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sotools` package."""


import unittest
#from click.testing import CliRunner

from sotools import enmap
from sotools import sharp
from sotools import curvedsky


class TestSotools(unittest.TestCase):
    """Tests for `sotools` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_000_something(self):
        """Test something."""
        pass

    # def test_command_line_interface(self):
    #     """Test the CLI."""
    #     runner = CliRunner()
    #     result = runner.invoke(cli.main)
    #     assert result.exit_code == 0
    #     assert 'sotools.cli.main' in result.output
    #     help_result = runner.invoke(cli.main, ['--help'])
    #     assert help_result.exit_code == 0
    #     assert '--help  Show this message and exit.' in help_result.output
