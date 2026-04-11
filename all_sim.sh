#!/bin/sh

echo "Run run.py"

python run.py --fov 60 120 180 240 300 360 --density 0.1 0.25 0.5 1.0 2.0 3.0 4.0 --L 40 --eta 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --v 0.5 --metric --topologic --workers 6

echo "Finish run.py"
