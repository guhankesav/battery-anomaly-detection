#!/bin/sh
# NASA Ames Prognostics Center of Excellence battery data (FY08 set: B0005, B0006, B0007, B0018), ~210 MB.
set -e
mkdir -p data/raw && cd data/raw
curl -fL -o nasa_battery.zip "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
unzip -q -o nasa_battery.zip -d x
unzip -q -o "x/5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip" -d fy08
