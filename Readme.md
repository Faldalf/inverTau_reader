# HORIBA InverTau FLIM HDF5 Viewer


Small Python viewer for quick inspection of FLIM datasets exported from a **HORIBA InverTau laser scanning microscope** as `.h5` / `.hdf5` files.

Prepared for a demo at the **University of Warwick, 22–23 September 2026**.

This tool is intended for **initial viewing, sanity checking, and quick exploratory analysis** only.
This tool supports export of the loaded file to .tiff (intensity and fastFLIM image) or tiff-stack (x,y,microtime). Any manipulation on the viewer does not affect exported raw data. 

## Features

- Intensity image
- HORIBA lifetime image
- Mean photon arrival image
- Overall and ROI TCSPC decays and phasors
- TIFF export of intensity and lifetime images
- Fiji/ImageJ-compatible FLIM TCSPC stack export

The FLIM histogram is stored internally as:

```text
Y × X × microtime
```

Summing over the microtime dimension reproduces the intensity image.

For Fiji/ImageJ export, the data are rearranged to:

```text
microtime × Y × X
```

so each stack frame is one spatial image at a single TCSPC bin.

Please note: At present it does not read FLIMera files (coming soon). 

## Installation

Download repository as .zip and extract or clone repository using git.

Use andaconda prompt, bash, terminal or similar.
Navigate to folder with the script and yml file (this repository).

```bash
cd C:\USers\whateverYourFilePathis\ 
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate horiba-flim-viewer
```

Run the viewer:

```bash
python inverTau-viewer.py
```

## Controls

**Load** — open a HORIBA `.h5` / `.hdf5` file.

**Intensity LUT / Threshold** — adjust display contrast. The Vmin value also acts as an intensity threshold for the FLIM and phasor displays.

**Linked FLIM LUT** — adjust Vmin/Vmax and select the colour map for both lifetime-style images.

**Phasor Filtering** — optional 1–5 pixel median filtering of the phasor G and S images only.

**Image ROI selection** — draw a spatial ROI and display its TCSPC and phasor distribution.

**Phasor ROI selection** — select a region in phasor space and highlight the corresponding image pixels.

**Export TIFF** — export raw intensity and HORIBA lifetime images.

**Export FLIM stack (Fiji)** — export the complete TCSPC cube as an ImageJ/Fiji-compatible time stack.

**Reset** — restore default LUTs, filters, ROIs, and zoom.

## Notes

The exported HORIBA files used during development contain accumulated FLIM data. Although metadata may report multiple acquisition frames, the individual frames are not retained in the exported HDF5 file.

The viewer does not currently perform IRF correction, phasor calibration, or full quantitative lifetime fitting.

## Disclaimer

This software is provided **as-is**, without warranty of any kind.

It is intended for demonstration and exploratory research use only. Users should validate results using appropriate calibration, reference measurements, and established analysis methods before drawing scientific conclusions.
