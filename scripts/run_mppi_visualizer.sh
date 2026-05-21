#!/bin/bash
exec python3 "$(rospack find dbf)/scripts/mppi_visualizer.py" "$@"
