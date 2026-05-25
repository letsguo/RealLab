#!/usr/bin/env python3
"""Launcher for mppi_visualizer.py that is safe to install via catkin_install_python.

This file is executed by the catkin-generated wrapper in devel/lib/dbf/, which
compiles and executes this file as Python code. Using a bash-style wrapper here
caused a SyntaxError when the wrapper attempted to compile it. Make this a
lightweight Python launcher that runs the real script.
"""
import runpy
import os
import sys

script_path = os.path.join(os.path.dirname(__file__), "mppi_visualizer.py")
sys.argv[0] = script_path
runpy.run_path(script_path, run_name="__main__")
