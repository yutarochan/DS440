#!/bin/bash
# Dataset Environment Setup Script
# Author: Yuya Jeremy Ong (yjo5006@psu.edu)

# Source: https://github.com/dfm/dr25/blob/master/get_data.sh

echo "[Dataset Environment Setup]\n"

# Create Directories
echo "> Generating Directories"
mkdir -p data
mkdir -p data/raw/
mkdir -p data/eda/

# Download KPLR DR25 PLTI Injected Dataset
echo "> Downloading: KPLR DR25 PLTI Injected Dataset"
mkdir -p data/raw/plti/
wget -O data/raw/plti/kplr_dr25_inj1_plti.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj1_plti.txt
wget -O data/raw/plti/kplr_dr25_inj2_plti.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj2_plti.txt
wget -O data/raw/plti/kplr_dr25_inj3_plti.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj3_plti.txt

# Download KPLR DR25 TCES Injected Dataset
echo "> Downloading: KPLR DR25 TCES Injected Dataset"
mkdir -p data/raw/tces/
wget -O data/raw/tces/kplr_dr25_inj1_tces.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj1_tces.txt
wget -O data/raw/tces/kplr_dr25_inj2_tces.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj2_tces.txt
wget -O data/raw/tces/kplr_dr25_inj3_tces.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj3_tces.txt

# Download KPLR DR25 Robovetter Injection Dataset
echo "> Downloading: KPLR DR25 Robovetter Injection Dataset"
git clone https://github.com/nasa/kepler-robovetter.git

# Build Robovetter Raw Data Directories
mkdir -p data/raw/robovetter/
mkdir -p data/raw/robovetter/inj
mkdir -p data/raw/robovetter/inv
mkdir -p data/raw/robovetter/obs
mkdir -p data/raw/robovetter/scr
mkdir -p data/raw/robovetter/sup

# Rearrange File Structure from Git Repository
mv kepler-robovetter/kplr_dr25_inj* data/raw/robovetter/inj
mv kepler-robovetter/kplr_dr25_inv* data/raw/robovetter/inv
mv kepler-robovetter/kplr_dr25_obs* data/raw/robovetter/obs
mv kepler-robovetter/kplr_dr25_scr* data/raw/robovetter/scr
mv kepler-robovetter/kplr_dr25_sup* data/raw/robovetter/sup

# Cleanup Directories
rm -rf kepler-robovetter

# Obtain Stellar and KOI Dataset
echo "> Download Stellar and KOI Dataset"
mkdir -p data/raw/misc/
wget -O data/raw/misc/q1_q17_dr25_stellar.txt "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_stellar&format=ipac&select=*"
wget -O data/raw/misc/q1_q17_dr25_koi.txt "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_koi&format=ipac&select=*"

echo "> Done"
