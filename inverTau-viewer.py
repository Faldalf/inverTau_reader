# inverTau-viewer
# small sample script with GUI to read hdf5 (h5) data exported from Horiba's inverTau platform. 
# Put together as part of a demo at the University of Warwick. 
# @faldalf | Falk Schneider 23/09/2026
#
# Please give it a try and see how it works for you
# Happy to try and support bug fixed and other requests. 
# Check readme for installation and usage. 


import sys
import os
import gc
import copy
import warnings

import h5py
import tifffile
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QGroupBox,
    QComboBox,
)

from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.widgets import LassoSelector

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

PHASOR_INTENSITY_THRESHOLD = 0


# ============================================================
# NUMPY-ONLY 2D MEDIAN FILTER
# ============================================================

def median_filter_2d_nan(data, size):
    """
    Spatial median filter using NumPy only.

    Applied ONLY to the phasor G and S images.

    Parameters
    ----------
    data : 2D ndarray

    size : int
        1 = no filtering
        2 = 2 x 2
        3 = 3 x 3
        4 = 4 x 4
        5 = 5 x 5
    """

    data = np.asarray(
        data,
        dtype=np.float64
    )

    if size <= 1:
        return data.copy()

    pad_before = size // 2

    pad_after = (
        size - 1 - pad_before
    )

    padded = np.pad(
        data,
        (
            (pad_before, pad_after),
            (pad_before, pad_after)
        ),
        mode="reflect"
    )

    windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            padded,
            (size, size)
        )
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning
        )

        filtered = np.nanmedian(
            windows,
            axis=(-2, -1)
        )

    return filtered


# ============================================================
# HORIBA HDF5 READER
# ============================================================

def read_horiba_flim_image_h5(filename):

    with h5py.File(filename, "r") as f:

        g = f["FLIM"]

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        attrs = {
            k: v
            for k, v in g.attrs.items()
        }

        width = int(
            attrs["Width"][0]
        )

        height = int(
            attrs["Height"][0]
        )

        nbins = int(
            attrs["NumberOfBins"][0]
        )

        time_per_bin = float(
            attrs["TimePerBin"][0]
        )

        # ----------------------------------------------------
        # Intensity
        # ----------------------------------------------------

        intensity = (
            g["Intensity"][()]
            .reshape(
                height,
                width
            )
        )

        # ----------------------------------------------------
        # HORIBA lifetime
        # ----------------------------------------------------

        lifetime = (
            g["Lifetime"][()]
            .reshape(
                height,
                width
            )
        )

        # For the files used during development, these HORIBA
        # lifetime values appear to correspond to TCSPC-bin
        # units.
        lifetime_ns = (
            lifetime
            * time_per_bin
            * 1e9
        )

        # ----------------------------------------------------
        # Per-pixel TCSPC histogram
        #
        # INTERNAL ARRAY ORDER:
        #
        #       Y x X x microtime
        #
        # histogram[y, x, :]
        # gives the decay at a single pixel.
        # ----------------------------------------------------

        histogram = (
            g["Histogram"][()]
            .reshape(
                height,
                width,
                nbins
            )
        )

        # ----------------------------------------------------
        # Global summed TCSPC
        # ----------------------------------------------------

        decay_sum = (
            g["Sum"][()][0]
        )

        # ----------------------------------------------------
        # HORIBA phasor
        # ----------------------------------------------------

        phasor_raw = (
            g["Phasor"][()]
        )

        phasor_x = (
            phasor_raw["X"]
            .reshape(
                height,
                width
            )
        )

        phasor_y = (
            phasor_raw["Y"]
            .reshape(
                height,
                width
            )
        )

    # ========================================================
    # TCSPC TIME AXIS
    # ========================================================

    # Bin centres
    t_s = (
        np.arange(nbins) + 0.5
    ) * time_per_bin

    t_ns = (
        t_s * 1e9
    )

    # ========================================================
    # MEAN PHOTON ARRIVAL TIME
    # ========================================================

    counts_per_pixel = (
        histogram.sum(
            axis=2
        )
    )

    mean_arrival_s = np.full(
        (height, width),
        np.nan,
        dtype=np.float64
    )

    valid = (
        counts_per_pixel > 0
    )

    mean_arrival_s[valid] = (
        (
            histogram[valid]
            * t_s
        )
        .sum(axis=1)
        / counts_per_pixel[valid]
    )

    mean_arrival_ns = (
        mean_arrival_s
        * 1e9
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "attrs": attrs,

        "width": width,
        "height": height,
        "nbins": nbins,

        "time_per_bin": time_per_bin,

        "t_ns": t_ns,

        "intensity": intensity,

        "lifetime": lifetime,
        "lifetime_ns": lifetime_ns,

        "histogram": histogram,

        "mean_arrival_ns": mean_arrival_ns,

        "decay_sum": decay_sum,

        "phasor_x": phasor_x,
        "phasor_y": phasor_y,
    }


# ============================================================
# GUI
# ============================================================

class HoribaViewer(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "HORIBA InverTau FLIM HDF5 Viewer - Version 05"
        )

        self.resize(
            1800,
            1080
        )

        # ====================================================
        # DATA
        # ====================================================

        self.img = None

        self.file_path = None
        self.file_name = None

        # ====================================================
        # IMAGE HANDLES
        # ====================================================

        self.im_intensity = None
        self.im_lifetime = None
        self.im_arrival = None

        self.cb_intensity = None
        self.cb_lifetime = None
        self.cb_arrival = None

        # ====================================================
        # LUT RANGES
        # ====================================================

        self.intensity_data_min = 0.0
        self.intensity_data_max = 1.0

        self.lifetime_data_min = 0.0
        self.lifetime_data_max = 1.0

        self.current_intensity_threshold = 0.0

        # ====================================================
        # FLIM COLOURMAP
        # ====================================================

        self.flim_cmap_name = "viridis"

        # ====================================================
        # PHASOR DISPLAY DATA
        # ====================================================

        self.phasor_x_display = None
        self.phasor_y_display = None

        self.phasor_filter_size = 1

        # ====================================================
        # IMAGE ROI
        # ====================================================

        self.image_roi_mask = None
        self.image_roi_vertices = None

        self.image_roi_patches = []

        # ====================================================
        # PHASOR ROI
        # ====================================================

        self.phasor_roi_mask = None
        self.phasor_roi_vertices = None

        self.phasor_roi_patch = None

        self.phasor_overlay_artists = []

        # ====================================================
        # LASSO SELECTORS
        # ====================================================

        self.lasso_selectors = []

        # ====================================================
        # MAIN WIDGET
        # ====================================================

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        # ====================================================
        # BUTTON ROW
        # ====================================================

        button_layout = QHBoxLayout()

        self.load_button = QPushButton(
            "Load"
        )

        self.clear_button = QPushButton(
            "Clear"
        )

        self.reset_button = QPushButton(
            "Reset"
        )

        self.export_button = QPushButton(
            "Export TIFF"
        )

        self.export_stack_button = QPushButton(
            "Export FLIM stack (Fiji)"
        )

        self.image_roi_button = QPushButton(
            "Image ROI selection"
        )

        self.image_roi_button.setCheckable(
            True
        )

        self.phasor_roi_button = QPushButton(
            "Phasor ROI selection"
        )

        self.phasor_roi_button.setCheckable(
            True
        )

        self.close_button = QPushButton(
            "Close"
        )

        button_layout.addWidget(
            self.load_button
        )

        button_layout.addWidget(
            self.clear_button
        )

        button_layout.addWidget(
            self.reset_button
        )

        button_layout.addWidget(
            self.export_button
        )

        button_layout.addWidget(
            self.export_stack_button
        )

        button_layout.addWidget(
            self.image_roi_button
        )

        button_layout.addWidget(
            self.phasor_roi_button
        )

        button_layout.addStretch()

        button_layout.addWidget(
            self.close_button
        )

        main_layout.addLayout(
            button_layout
        )

        # ====================================================
        # DISPLAY / FILTER CONTROL ROW
        # ====================================================

        controls_layout = QHBoxLayout()

        # ====================================================
        # INTENSITY LUT
        # ====================================================

        self.intensity_lut_group = QGroupBox(
            "Intensity LUT / Threshold"
        )

        intensity_lut_layout = QVBoxLayout(
            self.intensity_lut_group
        )

        # ----------------------------------------------------
        # Intensity Vmax
        # ----------------------------------------------------

        intensity_vmax_row = QHBoxLayout()

        self.intensity_vmax_label = QLabel(
            "Vmax: 100%"
        )

        self.intensity_vmax_label.setMinimumWidth(
            200
        )

        self.intensity_vmax_slider = QSlider(
            Qt.Horizontal
        )

        self.intensity_vmax_slider.setRange(
            1,
            100
        )

        self.intensity_vmax_slider.setValue(
            100
        )

        intensity_vmax_row.addWidget(
            self.intensity_vmax_label
        )

        intensity_vmax_row.addWidget(
            self.intensity_vmax_slider
        )

        intensity_lut_layout.addLayout(
            intensity_vmax_row
        )

        # ----------------------------------------------------
        # Intensity Vmin / threshold
        # ----------------------------------------------------

        intensity_vmin_row = QHBoxLayout()

        self.intensity_vmin_label = QLabel(
            "Vmin / threshold: 0%"
        )

        self.intensity_vmin_label.setMinimumWidth(
            200
        )

        self.intensity_vmin_slider = QSlider(
            Qt.Horizontal
        )

        self.intensity_vmin_slider.setRange(
            0,
            99
        )

        self.intensity_vmin_slider.setValue(
            0
        )

        intensity_vmin_row.addWidget(
            self.intensity_vmin_label
        )

        intensity_vmin_row.addWidget(
            self.intensity_vmin_slider
        )

        intensity_lut_layout.addLayout(
            intensity_vmin_row
        )

        # ====================================================
        # LINKED FLIM LUT
        # ====================================================

        self.lifetime_lut_group = QGroupBox(
            "Linked FLIM LUT"
        )

        lifetime_lut_layout = QVBoxLayout(
            self.lifetime_lut_group
        )

        # ----------------------------------------------------
        # FLIM Vmax
        # ----------------------------------------------------

        lifetime_vmax_row = QHBoxLayout()

        self.lifetime_vmax_label = QLabel(
            "Vmax: 100%"
        )

        self.lifetime_vmax_label.setMinimumWidth(
            185
        )

        self.lifetime_vmax_slider = QSlider(
            Qt.Horizontal
        )

        self.lifetime_vmax_slider.setRange(
            1,
            100
        )

        self.lifetime_vmax_slider.setValue(
            100
        )

        lifetime_vmax_row.addWidget(
            self.lifetime_vmax_label
        )

        lifetime_vmax_row.addWidget(
            self.lifetime_vmax_slider
        )

        lifetime_lut_layout.addLayout(
            lifetime_vmax_row
        )

        # ----------------------------------------------------
        # FLIM Vmin
        # ----------------------------------------------------

        lifetime_vmin_row = QHBoxLayout()

        self.lifetime_vmin_label = QLabel(
            "Vmin: 0%"
        )

        self.lifetime_vmin_label.setMinimumWidth(
            185
        )

        self.lifetime_vmin_slider = QSlider(
            Qt.Horizontal
        )

        self.lifetime_vmin_slider.setRange(
            0,
            99
        )

        self.lifetime_vmin_slider.setValue(
            0
        )

        lifetime_vmin_row.addWidget(
            self.lifetime_vmin_label
        )

        lifetime_vmin_row.addWidget(
            self.lifetime_vmin_slider
        )

        lifetime_lut_layout.addLayout(
            lifetime_vmin_row
        )

        # ----------------------------------------------------
        # Colour LUT
        # ----------------------------------------------------

        flim_cmap_row = QHBoxLayout()

        self.flim_cmap_label = QLabel(
            "Colour LUT:"
        )

        self.flim_cmap_label.setMinimumWidth(
            185
        )

        self.flim_cmap_combo = QComboBox()

        self.flim_cmap_combo.addItem(
            "Viridis",
            "viridis"
        )

        self.flim_cmap_combo.addItem(
            "Rainbow",
            "rainbow"
        )

        self.flim_cmap_combo.addItem(
            "Turbo",
            "turbo"
        )

        self.flim_cmap_combo.addItem(
            "Plasma",
            "plasma"
        )

        self.flim_cmap_combo.addItem(
            "Inferno",
            "inferno"
        )

        self.flim_cmap_combo.addItem(
            "Magma",
            "magma"
        )

        self.flim_cmap_combo.addItem(
            "Cividis",
            "cividis"
        )

        self.flim_cmap_combo.addItem(
            "Hot",
            "hot"
        )

        self.flim_cmap_combo.addItem(
            "Jet",
            "jet"
        )

        flim_cmap_row.addWidget(
            self.flim_cmap_label
        )

        flim_cmap_row.addWidget(
            self.flim_cmap_combo
        )

        lifetime_lut_layout.addLayout(
            flim_cmap_row
        )

        # ====================================================
        # PHASOR FILTER
        # ====================================================

        self.phasor_filter_group = QGroupBox(
            "Phasor Filtering"
        )

        phasor_filter_layout = QVBoxLayout(
            self.phasor_filter_group
        )

        phasor_filter_row = QHBoxLayout()

        self.phasor_filter_label = QLabel(
            "Median filter:"
        )

        self.phasor_filter_combo = QComboBox()

        self.phasor_filter_combo.addItem(
            "1 (no filter)",
            1
        )

        self.phasor_filter_combo.addItem(
            "2",
            2
        )

        self.phasor_filter_combo.addItem(
            "3",
            3
        )

        self.phasor_filter_combo.addItem(
            "4",
            4
        )

        self.phasor_filter_combo.addItem(
            "5",
            5
        )

        phasor_filter_row.addWidget(
            self.phasor_filter_label
        )

        phasor_filter_row.addWidget(
            self.phasor_filter_combo
        )

        phasor_filter_layout.addLayout(
            phasor_filter_row
        )

        self.phasor_filter_info = QLabel(
            "Filters G and S only"
        )

        phasor_filter_layout.addWidget(
            self.phasor_filter_info
        )

        # ====================================================
        # ADD CONTROLS
        # ====================================================

        controls_layout.addWidget(
            self.intensity_lut_group
        )

        controls_layout.addWidget(
            self.lifetime_lut_group
        )

        controls_layout.addWidget(
            self.phasor_filter_group
        )

        main_layout.addLayout(
            controls_layout
        )

        self.intensity_lut_group.setEnabled(
            False
        )

        self.lifetime_lut_group.setEnabled(
            False
        )

        self.phasor_filter_group.setEnabled(
            False
        )

        self.export_stack_button.setEnabled(
            False
        )

        # ====================================================
        # FILE INFO
        # ====================================================

        self.file_label = QLabel(
            "No file loaded"
        )

        main_layout.addWidget(
            self.file_label
        )

        # ====================================================
        # MATPLOTLIB
        # ====================================================

        self.fig = Figure(
            figsize=(17, 9)
        )

        self.canvas = FigureCanvas(
            self.fig
        )

        self.toolbar = NavigationToolbar(
            self.canvas,
            self
        )

        main_layout.addWidget(
            self.toolbar
        )

        main_layout.addWidget(
            self.canvas
        )

        # ====================================================
        # SIGNALS
        # ====================================================

        self.load_button.clicked.connect(
            self.load_file
        )

        self.clear_button.clicked.connect(
            self.clear_file
        )

        self.reset_button.clicked.connect(
            self.reset_view
        )

        self.export_button.clicked.connect(
            self.export_tiffs
        )

        self.export_stack_button.clicked.connect(
            self.export_flim_stack
        )

        self.image_roi_button.toggled.connect(
            self.toggle_image_roi
        )

        self.phasor_roi_button.toggled.connect(
            self.toggle_phasor_roi
        )

        self.intensity_vmax_slider.valueChanged.connect(
            self.update_intensity_lut
        )

        self.intensity_vmin_slider.valueChanged.connect(
            self.update_intensity_lut
        )

        self.lifetime_vmax_slider.valueChanged.connect(
            self.update_lifetime_lut
        )

        self.lifetime_vmin_slider.valueChanged.connect(
            self.update_lifetime_lut
        )

        self.flim_cmap_combo.currentIndexChanged.connect(
            self.update_flim_colormap
        )

        self.phasor_filter_combo.currentIndexChanged.connect(
            self.update_phasor_filter
        )

        self.close_button.clicked.connect(
            self.close
        )

        self.show_empty_figure()


    # ========================================================
    # AXIS STYLE
    # ========================================================

    def style_axis(self, ax):

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=13,
            width=2,
            length=6
        )

        for spine in ax.spines.values():

            spine.set_linewidth(2)


    # ========================================================
    # EMPTY FIGURE
    # ========================================================

    def show_empty_figure(self):

        self.fig.clear()

        ax = self.fig.add_subplot(
            111
        )

        ax.text(
            0.5,
            0.5,
            "Load a HORIBA FLIM HDF5 file",
            ha="center",
            va="center",
            fontsize=18,
            transform=ax.transAxes
        )

        ax.set_axis_off()

        self.canvas.draw_idle()


    # ========================================================
    # LOAD FILE
    # ========================================================

    def load_file(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open HORIBA HDF5 file",
            "",
            "HDF5 files (*.h5 *.hdf5);;"
            "All files (*)"
        )

        if not filename:
            return

        self.disable_lassos()

        self.img = None

        gc.collect()

        self.file_path = filename

        self.file_name = os.path.basename(
            filename
        )

        try:

            self.img = (
                read_horiba_flim_image_h5(
                    filename
                )
            )

        except Exception as exc:

            self.file_label.setText(
                f"Error loading file: {exc}"
            )

            self.show_empty_figure()

            return

        self.file_label.setText(
            f"{self.file_name}"
            f"    |    "
            f"{self.file_path}"
        )

        # ----------------------------------------------------
        # Clear selections
        # ----------------------------------------------------

        self.image_roi_mask = None
        self.image_roi_vertices = None
        self.image_roi_patches = []

        self.phasor_roi_mask = None
        self.phasor_roi_vertices = None
        self.phasor_roi_patch = None

        self.phasor_overlay_artists = []

        # ----------------------------------------------------
        # Reset phasor filter
        # ----------------------------------------------------

        self.phasor_filter_combo.blockSignals(
            True
        )

        self.phasor_filter_combo.setCurrentIndex(
            0
        )

        self.phasor_filter_combo.blockSignals(
            False
        )

        self.phasor_filter_size = 1

        self.phasor_x_display = (
            self.img["phasor_x"].astype(
                np.float64,
                copy=True
            )
        )

        self.phasor_y_display = (
            self.img["phasor_y"].astype(
                np.float64,
                copy=True
            )
        )

        # ----------------------------------------------------
        # Reset FLIM LUT
        # ----------------------------------------------------

        self.flim_cmap_combo.blockSignals(
            True
        )

        self.flim_cmap_combo.setCurrentIndex(
            0
        )

        self.flim_cmap_combo.blockSignals(
            False
        )

        self.flim_cmap_name = "viridis"

        # ----------------------------------------------------
        # Enable controls
        # ----------------------------------------------------

        self.intensity_lut_group.setEnabled(
            True
        )

        self.lifetime_lut_group.setEnabled(
            True
        )

        self.phasor_filter_group.setEnabled(
            True
        )

        self.export_stack_button.setEnabled(
            True
        )

        self.plot_data()


    # ========================================================
    # BUILD FIGURE
    # ========================================================

    def plot_data(self):

        if self.img is None:
            return

        self.disable_lassos()

        self.fig.clear()

        # ====================================================
        # MAIN GRID
        # ====================================================

        outer = self.fig.add_gridspec(
            2,
            1,
            height_ratios=[
                1.05,
                0.85
            ],
            hspace=0.38
        )

        # ====================================================
        # TOP GRID
        # ====================================================

        top = outer[0].subgridspec(
            1,
            3,
            wspace=0.32
        )

        # ----------------------------------------------------
        # Intensity
        # ----------------------------------------------------

        top_intensity = (
            top[0, 0].subgridspec(
                1,
                2,
                width_ratios=[
                    1.0,
                    0.055
                ],
                wspace=0.05
            )
        )

        self.ax_intensity = (
            self.fig.add_subplot(
                top_intensity[0, 0]
            )
        )

        self.cax_intensity = (
            self.fig.add_subplot(
                top_intensity[0, 1]
            )
        )

        # ----------------------------------------------------
        # Lifetime
        # ----------------------------------------------------

        top_lifetime = (
            top[0, 1].subgridspec(
                1,
                2,
                width_ratios=[
                    1.0,
                    0.055
                ],
                wspace=0.05
            )
        )

        self.ax_lifetime = (
            self.fig.add_subplot(
                top_lifetime[0, 0],
                sharex=self.ax_intensity,
                sharey=self.ax_intensity
            )
        )

        self.cax_lifetime = (
            self.fig.add_subplot(
                top_lifetime[0, 1]
            )
        )

        # ----------------------------------------------------
        # Mean arrival
        # ----------------------------------------------------

        top_arrival = (
            top[0, 2].subgridspec(
                1,
                2,
                width_ratios=[
                    1.0,
                    0.055
                ],
                wspace=0.05
            )
        )

        self.ax_arrival = (
            self.fig.add_subplot(
                top_arrival[0, 0],
                sharex=self.ax_intensity,
                sharey=self.ax_intensity
            )
        )

        self.cax_arrival = (
            self.fig.add_subplot(
                top_arrival[0, 1]
            )
        )

        for ax in [
            self.ax_intensity,
            self.ax_lifetime,
            self.ax_arrival,
        ]:

            ax.set_aspect(
                "equal",
                adjustable="box"
            )

        # ====================================================
        # BOTTOM GRID
        # ====================================================

        bottom = outer[1].subgridspec(
            1,
            4,
            wspace=0.36
        )

        self.ax_tcspc_all = (
            self.fig.add_subplot(
                bottom[0, 0]
            )
        )

        self.ax_tcspc_roi = (
            self.fig.add_subplot(
                bottom[0, 1]
            )
        )

        self.ax_phasor_all = (
            self.fig.add_subplot(
                bottom[0, 2]
            )
        )

        self.ax_phasor_roi = (
            self.fig.add_subplot(
                bottom[0, 3]
            )
        )

        # ====================================================
        # INTENSITY LUT RANGE
        # ====================================================

        intensity_data = (
            self.img["intensity"]
        )

        finite_intensity = intensity_data[
            np.isfinite(
                intensity_data
            )
        ]

        if finite_intensity.size > 0:

            self.intensity_data_min = float(
                np.min(
                    finite_intensity
                )
            )

            self.intensity_data_max = float(
                np.max(
                    finite_intensity
                )
            )

        else:

            self.intensity_data_min = 0.0
            self.intensity_data_max = 1.0

        # ====================================================
        # LINKED FLIM LUT RANGE
        # ====================================================

        lifetime_data = (
            self.img["lifetime_ns"]
        )

        arrival_data = (
            self.img["mean_arrival_ns"]
        )

        finite_lifetime = lifetime_data[
            np.isfinite(
                lifetime_data
            )
        ]

        finite_arrival = arrival_data[
            np.isfinite(
                arrival_data
            )
        ]

        if (
            finite_lifetime.size > 0
            or finite_arrival.size > 0
        ):

            combined = np.concatenate(
                (
                    finite_lifetime.ravel(),
                    finite_arrival.ravel()
                )
            )

            self.lifetime_data_min = float(
                np.min(
                    combined
                )
            )

            self.lifetime_data_max = float(
                np.max(
                    combined
                )
            )

        else:

            self.lifetime_data_min = 0.0
            self.lifetime_data_max = 1.0

        # ====================================================
        # RESET LUTS
        # ====================================================

        self.reset_lut_sliders()

        self.current_intensity_threshold = (
            self.intensity_data_min
        )

        # ====================================================
        # INTENSITY IMAGE
        # ====================================================

        self.im_intensity = (
            self.ax_intensity.imshow(
                self.img["intensity"],
                cmap="gray",
                interpolation="nearest",
                aspect="equal",
                vmin=self.intensity_data_min,
                vmax=self.intensity_data_max
            )
        )

        self.ax_intensity.set_title(
            "Intensity",
            fontsize=16
        )

        self.ax_intensity.set_xlabel(
            "X (pixel)",
            fontsize=16
        )

        self.ax_intensity.set_ylabel(
            "Y (pixel)",
            fontsize=16
        )

        self.cb_intensity = (
            self.fig.colorbar(
                self.im_intensity,
                cax=self.cax_intensity
            )
        )

        self.cb_intensity.ax.tick_params(
            labelsize=12,
            width=2
        )

        # ====================================================
        # FLIM COLORMAP
        # ====================================================

        lifetime_cmap = (
            self.get_flim_colormap()
        )

        # ====================================================
        # HORIBA LIFETIME
        # ====================================================

        lifetime_display = (
            self.make_thresholded_image(
                self.img["lifetime_ns"]
            )
        )

        self.im_lifetime = (
            self.ax_lifetime.imshow(
                lifetime_display,
                cmap=lifetime_cmap,
                interpolation="nearest",
                aspect="equal",
                vmin=self.lifetime_data_min,
                vmax=self.lifetime_data_max
            )
        )

        self.ax_lifetime.set_title(
            "HORIBA lifetime",
            fontsize=16
        )

        self.ax_lifetime.set_xlabel(
            "X (pixel)",
            fontsize=16
        )

        self.ax_lifetime.set_ylabel(
            "Y (pixel)",
            fontsize=16
        )

        self.cb_lifetime = (
            self.fig.colorbar(
                self.im_lifetime,
                cax=self.cax_lifetime
            )
        )

        self.cb_lifetime.set_label(
            "Lifetime (ns)",
            fontsize=13,
            labelpad=6
        )

        self.cb_lifetime.ax.tick_params(
            labelsize=12,
            width=2
        )

        # ====================================================
        # MEAN ARRIVAL
        # ====================================================

        arrival_display = (
            self.make_thresholded_image(
                self.img["mean_arrival_ns"]
            )
        )

        self.im_arrival = (
            self.ax_arrival.imshow(
                arrival_display,
                cmap=lifetime_cmap,
                interpolation="nearest",
                aspect="equal",
                vmin=self.lifetime_data_min,
                vmax=self.lifetime_data_max
            )
        )

        self.ax_arrival.set_title(
            "Mean photon arrival",
            fontsize=16
        )

        self.ax_arrival.set_xlabel(
            "X (pixel)",
            fontsize=16
        )

        self.ax_arrival.set_ylabel(
            "Y (pixel)",
            fontsize=16
        )

        self.cb_arrival = (
            self.fig.colorbar(
                self.im_arrival,
                cax=self.cax_arrival
            )
        )

        self.cb_arrival.set_label(
            "Mean arrival (ns)",
            fontsize=13,
            labelpad=6
        )

        self.cb_arrival.ax.tick_params(
            labelsize=12,
            width=2
        )

        # ====================================================
        # OVERALL TCSPC
        # ====================================================

        self.ax_tcspc_all.semilogy(
            self.img["t_ns"],
            self.img["decay_sum"],
            linewidth=2
        )

        self.ax_tcspc_all.set_title(
            "Overall TCSPC",
            fontsize=16
        )

        self.ax_tcspc_all.set_xlabel(
            "Time (ns)",
            fontsize=16
        )

        self.ax_tcspc_all.set_ylabel(
            "Counts",
            fontsize=16
        )

        self.reset_roi_tcspc_axis()

        self.draw_overall_phasor()

        self.reset_roi_phasor_axis()

        # ====================================================
        # STYLE
        # ====================================================

        for ax in [
            self.ax_intensity,
            self.ax_lifetime,
            self.ax_arrival,
            self.ax_tcspc_all,
            self.ax_tcspc_roi,
            self.ax_phasor_all,
            self.ax_phasor_roi,
        ]:

            self.style_axis(
                ax
            )

        self.fig.subplots_adjust(
            left=0.055,
            right=0.975,
            bottom=0.07,
            top=0.95
        )

        self.reset_image_zoom()

        self.update_lut_labels()

        self.canvas.draw_idle()


    # ========================================================
    # GET FLIM COLORMAP
    # ========================================================

    def get_flim_colormap(self):

        try:

            cmap = copy.copy(
                plt.get_cmap(
                    self.flim_cmap_name
                )
            )

        except Exception:

            self.flim_cmap_name = "viridis"

            cmap = copy.copy(
                plt.get_cmap(
                    "viridis"
                )
            )

        # Masked pixels are black
        cmap.set_bad(
            color="black"
        )

        return cmap


    # ========================================================
    # UPDATE FLIM COLORMAP
    # ========================================================

    def update_flim_colormap(self, *_):

        if self.img is None:
            return

        cmap_name = (
            self.flim_cmap_combo.currentData()
        )

        if cmap_name is None:

            cmap_name = "viridis"

        self.flim_cmap_name = str(
            cmap_name
        )

        cmap = (
            self.get_flim_colormap()
        )

        if self.im_lifetime is not None:

            self.im_lifetime.set_cmap(
                cmap
            )

        if self.im_arrival is not None:

            self.im_arrival.set_cmap(
                cmap
            )

        if (
            self.cb_lifetime is not None
            and self.im_lifetime is not None
        ):

            self.cb_lifetime.update_normal(
                self.im_lifetime
            )

        if (
            self.cb_arrival is not None
            and self.im_arrival is not None
        ):

            self.cb_arrival.update_normal(
                self.im_arrival
            )

        self.canvas.draw_idle()


    # ========================================================
    # PERCENT -> DATA VALUE
    # ========================================================

    def percent_to_value(
        self,
        percent,
        data_min,
        data_max
    ):

        return (
            data_min
            + (
                percent / 100.0
            )
            * (
                data_max
                - data_min
            )
        )


    # ========================================================
    # RESET LUT SLIDERS
    # ========================================================

    def reset_lut_sliders(self):

        sliders = [
            self.intensity_vmax_slider,
            self.intensity_vmin_slider,
            self.lifetime_vmax_slider,
            self.lifetime_vmin_slider,
        ]

        for slider in sliders:

            slider.blockSignals(
                True
            )

        self.intensity_vmax_slider.setValue(
            100
        )

        self.intensity_vmin_slider.setValue(
            0
        )

        self.lifetime_vmax_slider.setValue(
            100
        )

        self.lifetime_vmin_slider.setValue(
            0
        )

        for slider in sliders:

            slider.blockSignals(
                False
            )


    # ========================================================
    # MAKE THRESHOLDED FLIM IMAGE
    # ========================================================

    def make_thresholded_image(
        self,
        data
    ):

        if self.img is None:
            return data

        intensity = (
            self.img["intensity"]
        )

        threshold = (
            self.current_intensity_threshold
        )

        mask = (
            (intensity < threshold)
            | ~np.isfinite(data)
        )

        return np.ma.array(
            data,
            mask=mask
        )


    # ========================================================
    # APPLY INTENSITY THRESHOLD TO FLIM IMAGES
    # ========================================================

    def apply_intensity_threshold_to_images(self):

        if (
            self.img is None
            or self.im_lifetime is None
            or self.im_arrival is None
        ):

            return

        lifetime_display = (
            self.make_thresholded_image(
                self.img["lifetime_ns"]
            )
        )

        arrival_display = (
            self.make_thresholded_image(
                self.img["mean_arrival_ns"]
            )
        )

        self.im_lifetime.set_data(
            lifetime_display
        )

        self.im_arrival.set_data(
            arrival_display
        )


    # ========================================================
    # INTENSITY LUT + THRESHOLD
    # ========================================================

    def update_intensity_lut(self, *_):

        if (
            self.img is None
            or self.im_intensity is None
        ):

            return

        old_threshold = (
            self.current_intensity_threshold
        )

        vmin_percent = (
            self.intensity_vmin_slider.value()
        )

        vmax_percent = (
            self.intensity_vmax_slider.value()
        )

        sender = self.sender()

        # ----------------------------------------------------
        # Prevent Vmin >= Vmax
        # ----------------------------------------------------

        if vmin_percent >= vmax_percent:

            if (
                sender
                is self.intensity_vmin_slider
            ):

                new_vmax = min(
                    100,
                    vmin_percent + 1
                )

                self.intensity_vmax_slider.blockSignals(
                    True
                )

                self.intensity_vmax_slider.setValue(
                    new_vmax
                )

                self.intensity_vmax_slider.blockSignals(
                    False
                )

                vmax_percent = new_vmax

            else:

                new_vmin = max(
                    0,
                    vmax_percent - 1
                )

                self.intensity_vmin_slider.blockSignals(
                    True
                )

                self.intensity_vmin_slider.setValue(
                    new_vmin
                )

                self.intensity_vmin_slider.blockSignals(
                    False
                )

                vmin_percent = new_vmin

        vmin = self.percent_to_value(
            vmin_percent,
            self.intensity_data_min,
            self.intensity_data_max
        )

        vmax = self.percent_to_value(
            vmax_percent,
            self.intensity_data_min,
            self.intensity_data_max
        )

        self.im_intensity.set_clim(
            vmin=vmin,
            vmax=vmax
        )

        if self.cb_intensity is not None:

            self.cb_intensity.update_normal(
                self.im_intensity
            )

        # Vmin is also threshold
        self.current_intensity_threshold = (
            vmin
        )

        self.apply_intensity_threshold_to_images()

        self.update_lut_labels()

        threshold_changed = not np.isclose(
            old_threshold,
            self.current_intensity_threshold,
            rtol=0,
            atol=1e-12
        )

        if threshold_changed:

            self.refresh_phasor_views()

        self.canvas.draw_idle()


    # ========================================================
    # LINKED FLIM LUT
    # ========================================================

    def update_lifetime_lut(self, *_):

        if (
            self.img is None
            or self.im_lifetime is None
            or self.im_arrival is None
        ):

            return

        vmin_percent = (
            self.lifetime_vmin_slider.value()
        )

        vmax_percent = (
            self.lifetime_vmax_slider.value()
        )

        sender = self.sender()

        if vmin_percent >= vmax_percent:

            if (
                sender
                is self.lifetime_vmin_slider
            ):

                new_vmax = min(
                    100,
                    vmin_percent + 1
                )

                self.lifetime_vmax_slider.blockSignals(
                    True
                )

                self.lifetime_vmax_slider.setValue(
                    new_vmax
                )

                self.lifetime_vmax_slider.blockSignals(
                    False
                )

                vmax_percent = new_vmax

            else:

                new_vmin = max(
                    0,
                    vmax_percent - 1
                )

                self.lifetime_vmin_slider.blockSignals(
                    True
                )

                self.lifetime_vmin_slider.setValue(
                    new_vmin
                )

                self.lifetime_vmin_slider.blockSignals(
                    False
                )

                vmin_percent = new_vmin

        vmin = self.percent_to_value(
            vmin_percent,
            self.lifetime_data_min,
            self.lifetime_data_max
        )

        vmax = self.percent_to_value(
            vmax_percent,
            self.lifetime_data_min,
            self.lifetime_data_max
        )

        self.im_lifetime.set_clim(
            vmin=vmin,
            vmax=vmax
        )

        self.im_arrival.set_clim(
            vmin=vmin,
            vmax=vmax
        )

        if self.cb_lifetime is not None:

            self.cb_lifetime.update_normal(
                self.im_lifetime
            )

        if self.cb_arrival is not None:

            self.cb_arrival.update_normal(
                self.im_arrival
            )

        self.update_lut_labels()

        self.canvas.draw_idle()


    # ========================================================
    # LUT LABELS
    # ========================================================

    def update_lut_labels(self):

        # Intensity
        i_min_pct = (
            self.intensity_vmin_slider.value()
        )

        i_max_pct = (
            self.intensity_vmax_slider.value()
        )

        i_min = self.percent_to_value(
            i_min_pct,
            self.intensity_data_min,
            self.intensity_data_max
        )

        i_max = self.percent_to_value(
            i_max_pct,
            self.intensity_data_min,
            self.intensity_data_max
        )

        self.intensity_vmin_label.setText(
            f"Vmin / threshold: "
            f"{i_min_pct}%  "
            f"({i_min:.3g})"
        )

        self.intensity_vmax_label.setText(
            f"Vmax: "
            f"{i_max_pct}%  "
            f"({i_max:.3g})"
        )

        # Lifetime
        l_min_pct = (
            self.lifetime_vmin_slider.value()
        )

        l_max_pct = (
            self.lifetime_vmax_slider.value()
        )

        l_min = self.percent_to_value(
            l_min_pct,
            self.lifetime_data_min,
            self.lifetime_data_max
        )

        l_max = self.percent_to_value(
            l_max_pct,
            self.lifetime_data_min,
            self.lifetime_data_max
        )

        self.lifetime_vmin_label.setText(
            f"Vmin: "
            f"{l_min_pct}%  "
            f"({l_min:.3f} ns)"
        )

        self.lifetime_vmax_label.setText(
            f"Vmax: "
            f"{l_max_pct}%  "
            f"({l_max:.3f} ns)"
        )


    # ========================================================
    # PHASOR MEDIAN FILTER
    # ========================================================

    def update_phasor_filter(self, *_):

        if self.img is None:
            return

        size = (
            self.phasor_filter_combo.currentData()
        )

        if size is None:
            size = 1

        size = int(
            size
        )

        self.phasor_filter_size = (
            size
        )

        if size <= 1:

            self.phasor_x_display = (
                self.img["phasor_x"]
                .astype(
                    np.float64,
                    copy=True
                )
            )

            self.phasor_y_display = (
                self.img["phasor_y"]
                .astype(
                    np.float64,
                    copy=True
                )
            )

        else:

            self.phasor_x_display = (
                median_filter_2d_nan(
                    self.img["phasor_x"],
                    size
                )
            )

            self.phasor_y_display = (
                median_filter_2d_nan(
                    self.img["phasor_y"],
                    size
                )
            )

        self.refresh_phasor_views()

        self.canvas.draw_idle()


    # ========================================================
    # GET CURRENT PHASOR ARRAYS
    # ========================================================

    def get_phasor_arrays(self):

        if self.phasor_x_display is None:

            G = self.img[
                "phasor_x"
            ]

        else:

            G = (
                self.phasor_x_display
            )

        if self.phasor_y_display is None:

            S = self.img[
                "phasor_y"
            ]

        else:

            S = (
                self.phasor_y_display
            )

        return G, S


    # ========================================================
    # VALID PHASOR MASK
    # ========================================================

    def get_valid_phasor_mask(self):

        G, S = (
            self.get_phasor_arrays()
        )

        intensity = (
            self.img["intensity"]
        )

        threshold = max(
            float(
                self.current_intensity_threshold
            ),
            float(
                PHASOR_INTENSITY_THRESHOLD
            )
        )

        valid = (
            (intensity >= threshold)
            & (intensity > PHASOR_INTENSITY_THRESHOLD)
            & np.isfinite(G)
            & np.isfinite(S)
        )

        return valid


    # ========================================================
    # UNIVERSAL PHASOR SEMICIRCLE
    # ========================================================

    def universal_semicircle(self):

        theta = np.linspace(
            0,
            np.pi,
            500
        )

        G = (
            0.5
            + 0.5 * np.cos(theta)
        )

        S = (
            0.5
            * np.sin(theta)
        )

        return G, S


    # ========================================================
    # PHASOR AXIS SETUP
    # ========================================================

    def setup_phasor_axis(
        self,
        ax,
        title
    ):

        ax.clear()

        G_uc, S_uc = (
            self.universal_semicircle()
        )

        ax.plot(
            G_uc,
            S_uc,
            linewidth=2
        )

        ax.set_title(
            title,
            fontsize=16
        )

        ax.set_xlabel(
            "G",
            fontsize=16
        )

        ax.set_ylabel(
            "S",
            fontsize=16
        )

        ax.set_xlim(
            -0.05,
            1.05
        )

        ax.set_ylim(
            -0.05,
            0.60
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )


    # ========================================================
    # DRAW OVERALL PHASOR
    # ========================================================

    def draw_overall_phasor(self):

        filter_text = (
            "unfiltered"
            if self.phasor_filter_size == 1
            else
            f"{self.phasor_filter_size}x"
            f"{self.phasor_filter_size} median"
        )

        self.setup_phasor_axis(
            self.ax_phasor_all,
            f"Overall phasor ({filter_text})"
        )

        self.phasor_roi_patch = None

        G, S = (
            self.get_phasor_arrays()
        )

        valid = (
            self.get_valid_phasor_mask()
        )

        self.ax_phasor_all.scatter(
            G[valid],
            S[valid],
            s=2,
            alpha=0.25,
            rasterized=True
        )

        self.style_axis(
            self.ax_phasor_all
        )


    # ========================================================
    # RESET ROI TCSPC
    # ========================================================

    def reset_roi_tcspc_axis(self):

        self.ax_tcspc_roi.clear()

        self.ax_tcspc_roi.set_title(
            "Image ROI TCSPC",
            fontsize=16
        )

        self.ax_tcspc_roi.set_xlabel(
            "Time (ns)",
            fontsize=16
        )

        self.ax_tcspc_roi.set_ylabel(
            "Counts",
            fontsize=16
        )

        self.ax_tcspc_roi.text(
            0.5,
            0.5,
            "No image ROI selected",
            ha="center",
            va="center",
            fontsize=13,
            transform=self.ax_tcspc_roi.transAxes
        )

        self.style_axis(
            self.ax_tcspc_roi
        )


    # ========================================================
    # RESET ROI PHASOR
    # ========================================================

    def reset_roi_phasor_axis(self):

        self.setup_phasor_axis(
            self.ax_phasor_roi,
            "Image ROI phasor"
        )

        self.ax_phasor_roi.text(
            0.5,
            0.5,
            "No image ROI selected",
            ha="center",
            va="center",
            fontsize=13,
            transform=self.ax_phasor_roi.transAxes
        )

        self.style_axis(
            self.ax_phasor_roi
        )


    # ========================================================
    # DRAW IMAGE ROI PHASOR
    # ========================================================

    def draw_image_roi_phasor(self):

        if self.image_roi_mask is None:

            self.reset_roi_phasor_axis()

            return

        n_pixels = int(
            self.image_roi_mask.sum()
        )

        self.setup_phasor_axis(
            self.ax_phasor_roi,
            f"Image ROI phasor "
            f"({n_pixels} pixels)"
        )

        G, S = (
            self.get_phasor_arrays()
        )

        valid = (
            self.get_valid_phasor_mask()
        )

        mask = (
            self.image_roi_mask
            & valid
        )

        self.ax_phasor_roi.scatter(
            G[mask],
            S[mask],
            s=4,
            alpha=0.35,
            rasterized=True
        )

        self.style_axis(
            self.ax_phasor_roi
        )


    # ========================================================
    # REFRESH PHASOR VIEWS
    # ========================================================

    def refresh_phasor_views(self):

        if self.img is None:
            return

        self.draw_overall_phasor()

        if self.image_roi_mask is not None:

            self.draw_image_roi_phasor()

        else:

            self.reset_roi_phasor_axis()

        if self.phasor_roi_vertices is not None:

            self.recompute_phasor_roi_mask()

            self.draw_phasor_roi_outline()

            self.update_phasor_image_overlay()

        else:

            self.clear_phasor_overlay()

        self.canvas.draw_idle()


    # ========================================================
    # BUTTON STATE WITHOUT SIGNAL
    # ========================================================

    def set_button_without_signal(
        self,
        button,
        value
    ):

        button.blockSignals(
            True
        )

        button.setChecked(
            value
        )

        button.blockSignals(
            False
        )


    # ========================================================
    # IMAGE ROI TOGGLE
    # ========================================================

    def toggle_image_roi(
        self,
        checked
    ):

        if self.img is None:

            self.set_button_without_signal(
                self.image_roi_button,
                False
            )

            return

        if checked:

            self.set_button_without_signal(
                self.phasor_roi_button,
                False
            )

            self.disable_lassos()

            self.disable_toolbar_navigation()

            self.enable_image_lasso()

        else:

            self.disable_lassos()


    # ========================================================
    # PHASOR ROI TOGGLE
    # ========================================================

    def toggle_phasor_roi(
        self,
        checked
    ):

        if self.img is None:

            self.set_button_without_signal(
                self.phasor_roi_button,
                False
            )

            return

        if checked:

            self.set_button_without_signal(
                self.image_roi_button,
                False
            )

            self.disable_lassos()

            self.disable_toolbar_navigation()

            self.reset_image_zoom()

            self.enable_phasor_lasso()

        else:

            self.disable_lassos()


    # ========================================================
    # DISABLE TOOLBAR PAN / ZOOM
    # ========================================================

    def disable_toolbar_navigation(self):

        mode = str(
            self.toolbar.mode
        ).lower()

        if "pan" in mode:

            self.toolbar.pan()

        elif "zoom" in mode:

            self.toolbar.zoom()


    # ========================================================
    # ENABLE IMAGE LASSO
    # ========================================================

    def enable_image_lasso(self):

        for ax in [
            self.ax_intensity,
            self.ax_lifetime,
            self.ax_arrival,
        ]:

            selector = LassoSelector(
                ax,
                onselect=self.on_image_lasso,
                useblit=True,
                button=1,
                props={
                    "linewidth": 2
                }
            )

            self.lasso_selectors.append(
                selector
            )


    # ========================================================
    # ENABLE PHASOR LASSO
    # ========================================================

    def enable_phasor_lasso(self):

        selector = LassoSelector(
            self.ax_phasor_all,
            onselect=self.on_phasor_lasso,
            useblit=True,
            button=1,
            props={
                "linewidth": 2
            }
        )

        self.lasso_selectors.append(
            selector
        )


    # ========================================================
    # DISABLE ALL LASSOS
    # ========================================================

    def disable_lassos(self):

        for selector in (
            self.lasso_selectors
        ):

            try:

                selector.set_active(
                    False
                )

                selector.disconnect_events()

            except Exception:

                pass

        self.lasso_selectors = []


    # ========================================================
    # IMAGE LASSO CALLBACK
    # ========================================================

    def on_image_lasso(
        self,
        vertices
    ):

        if self.img is None:
            return

        if len(vertices) < 3:
            return

        vertices = np.asarray(
            vertices
        )

        height = (
            self.img["height"]
        )

        width = (
            self.img["width"]
        )

        yy, xx = np.indices(
            (height, width)
        )

        points = np.column_stack(
            (
                xx.ravel(),
                yy.ravel()
            )
        )

        path = Path(
            vertices
        )

        mask = (
            path.contains_points(
                points
            )
            .reshape(
                height,
                width
            )
        )

        if not np.any(mask):
            return

        self.image_roi_vertices = (
            vertices
        )

        self.image_roi_mask = (
            mask
        )

        self.draw_image_roi_outline()

        self.update_image_roi_plots()


    # ========================================================
    # DRAW IMAGE ROI OUTLINE
    # ========================================================

    def draw_image_roi_outline(self):

        for patch in (
            self.image_roi_patches
        ):

            try:

                patch.remove()

            except Exception:

                pass

        self.image_roi_patches = []

        if self.image_roi_vertices is None:
            return

        for ax in [
            self.ax_intensity,
            self.ax_lifetime,
            self.ax_arrival,
        ]:

            patch = Polygon(
                self.image_roi_vertices,
                closed=True,
                fill=False,
                linewidth=2
            )

            ax.add_patch(
                patch
            )

            self.image_roi_patches.append(
                patch
            )

        self.canvas.draw_idle()


    # ========================================================
    # UPDATE IMAGE ROI PLOTS
    # ========================================================

    def update_image_roi_plots(self):

        if self.image_roi_mask is None:
            return

        n_pixels = int(
            self.image_roi_mask.sum()
        )

        roi_tcspc = (
            self.img["histogram"][
                self.image_roi_mask
            ]
            .sum(axis=0)
        )

        self.ax_tcspc_roi.clear()

        self.ax_tcspc_roi.semilogy(
            self.img["t_ns"],
            roi_tcspc,
            linewidth=2
        )

        self.ax_tcspc_roi.set_title(
            f"Image ROI TCSPC "
            f"({n_pixels} pixels)",
            fontsize=16
        )

        self.ax_tcspc_roi.set_xlabel(
            "Time (ns)",
            fontsize=16
        )

        self.ax_tcspc_roi.set_ylabel(
            "Counts",
            fontsize=16
        )

        self.style_axis(
            self.ax_tcspc_roi
        )

        self.draw_image_roi_phasor()

        self.canvas.draw_idle()


    # ========================================================
    # PHASOR LASSO CALLBACK
    # ========================================================

    def on_phasor_lasso(
        self,
        vertices
    ):

        if self.img is None:
            return

        if len(vertices) < 3:
            return

        self.phasor_roi_vertices = (
            np.asarray(
                vertices
            )
        )

        self.recompute_phasor_roi_mask()

        self.draw_phasor_roi_outline()

        self.update_phasor_image_overlay()


    # ========================================================
    # RECOMPUTE PHASOR ROI MASK
    # ========================================================

    def recompute_phasor_roi_mask(self):

        if (
            self.img is None
            or self.phasor_roi_vertices is None
        ):

            self.phasor_roi_mask = None
            return

        G, S = (
            self.get_phasor_arrays()
        )

        valid = (
            self.get_valid_phasor_mask()
        )

        mask_flat = np.zeros(
            G.size,
            dtype=bool
        )

        valid_flat = (
            valid.ravel()
        )

        points = np.column_stack(
            (
                G.ravel()[
                    valid_flat
                ],
                S.ravel()[
                    valid_flat
                ]
            )
        )

        if points.size == 0:

            self.phasor_roi_mask = (
                mask_flat.reshape(
                    G.shape
                )
            )

            return

        path = Path(
            self.phasor_roi_vertices
        )

        inside = (
            path.contains_points(
                points
            )
        )

        mask_flat[
            valid_flat
        ] = inside

        self.phasor_roi_mask = (
            mask_flat.reshape(
                G.shape
            )
        )


    # ========================================================
    # DRAW PHASOR ROI
    # ========================================================

    def draw_phasor_roi_outline(self):

        if (
            self.phasor_roi_patch
            is not None
        ):

            try:

                self.phasor_roi_patch.remove()

            except Exception:

                pass

            self.phasor_roi_patch = None

        if self.phasor_roi_vertices is None:
            return

        self.phasor_roi_patch = Polygon(
            self.phasor_roi_vertices,
            closed=True,
            fill=False,
            linewidth=2,
            edgecolor="red"
        )

        self.ax_phasor_all.add_patch(
            self.phasor_roi_patch
        )


    # ========================================================
    # CLEAR PHASOR OVERLAY
    # ========================================================

    def clear_phasor_overlay(self):

        for artist in (
            self.phasor_overlay_artists
        ):

            try:

                artist.remove()

            except Exception:

                pass

        self.phasor_overlay_artists = []


    # ========================================================
    # PHASOR SELECTION -> IMAGE OVERLAY
    # ========================================================

    def update_phasor_image_overlay(self):

        self.clear_phasor_overlay()

        if self.phasor_roi_mask is None:
            return

        height = (
            self.img["height"]
        )

        width = (
            self.img["width"]
        )

        rgba = np.zeros(
            (height, width, 4),
            dtype=np.float32
        )

        rgba[..., 0] = 1.0

        rgba[..., 3] = (
            self.phasor_roi_mask.astype(
                np.float32
            )
            * 0.60
        )

        old_xlim = (
            self.ax_intensity.get_xlim()
        )

        old_ylim = (
            self.ax_intensity.get_ylim()
        )

        for ax in [
            self.ax_intensity,
            self.ax_lifetime,
            self.ax_arrival,
        ]:

            artist = ax.imshow(
                rgba,
                interpolation="nearest",
                aspect="equal",
                zorder=10
            )

            ax.set_aspect(
                "equal",
                adjustable="box"
            )

            self.phasor_overlay_artists.append(
                artist
            )

        self.ax_intensity.set_xlim(
            old_xlim
        )

        self.ax_intensity.set_ylim(
            old_ylim
        )

        n_pixels = int(
            self.phasor_roi_mask.sum()
        )

        filter_text = (
            "unfiltered"
            if self.phasor_filter_size == 1
            else
            f"{self.phasor_filter_size}x"
            f"{self.phasor_filter_size} median"
        )

        self.ax_phasor_all.set_title(
            f"Overall phasor "
            f"({filter_text}, "
            f"{n_pixels} selected)",
            fontsize=16
        )

        self.canvas.draw_idle()


    # ========================================================
    # RESET IMAGE ZOOM
    # ========================================================

    def reset_image_zoom(self):

        if self.img is None:
            return

        width = (
            self.img["width"]
        )

        height = (
            self.img["height"]
        )

        self.ax_intensity.set_xlim(
            -0.5,
            width - 0.5
        )

        self.ax_intensity.set_ylim(
            height - 0.5,
            -0.5
        )

        for ax in [
            self.ax_intensity,
            self.ax_lifetime,
            self.ax_arrival,
        ]:

            ax.set_aspect(
                "equal",
                adjustable="box"
            )


    # ========================================================
    # EXPORT ORDINARY TIFFS
    # ========================================================

    def export_tiffs(self):

        if self.img is None:
            return

        if self.file_path is not None:

            folder = os.path.dirname(
                self.file_path
            )

            stem = os.path.splitext(
                os.path.basename(
                    self.file_path
                )
            )[0]

            default_path = os.path.join(
                folder,
                stem + ".tif"
            )

        else:

            default_path = (
                "horiba_export.tif"
            )

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export TIFF images",
            default_path,
            "TIFF files (*.tif *.tiff)"
        )

        if not filename:
            return

        base, _ = os.path.splitext(
            filename
        )

        intensity_filename = (
            base
            + "_intensity.tif"
        )

        lifetime_filename = (
            base
            + "_lifetime_ns.tif"
        )

        try:

            tifffile.imwrite(
                intensity_filename,
                np.asarray(
                    self.img["intensity"]
                ),
                metadata={
                    "axes": "YX"
                }
            )

            tifffile.imwrite(
                lifetime_filename,
                np.asarray(
                    self.img["lifetime_ns"],
                    dtype=np.float32
                ),
                metadata={
                    "axes": "YX",
                    "unit": "ns"
                }
            )

        except Exception as exc:

            self.file_label.setText(
                f"TIFF export error: {exc}"
            )

            return

        self.file_label.setText(
            f"Exported: "
            f"{os.path.basename(intensity_filename)}"
            f"    |    "
            f"{os.path.basename(lifetime_filename)}"
        )


    # ========================================================
    # EXPORT FLIM STACK FOR FIJI / IMAGEJ
    # ========================================================

    def export_flim_stack(self):
        """
        Export the complete per-pixel TCSPC cube in an
        ImageJ/Fiji-friendly arrangement.

        Internal HORIBA representation:

            histogram[Y, X, T]

        For Fiji the data are rearranged to:

            stack[T, Y, X]

        Therefore each TIFF plane is a normal spatial image
        corresponding to ONE microtime bin.

        In Fiji:
            moving through the T/frame slider moves through
            the TCSPC / microtime dimension.

        Summing across T returns the intensity image:

            stack.sum(axis=0)

        Raw TCSPC counts are exported. Display thresholds,
        colour LUTs, ROI selections and phasor filtering do
        not modify the exported data.
        """

        if self.img is None:
            return

        # ----------------------------------------------------
        # Default filename
        # ----------------------------------------------------

        if self.file_path is not None:

            folder = os.path.dirname(
                self.file_path
            )

            stem = os.path.splitext(
                os.path.basename(
                    self.file_path
                )
            )[0]

            default_path = os.path.join(
                folder,
                stem
                + "_FLIM_TCSPC_Fiji.tif"
            )

        else:

            default_path = (
                "FLIM_TCSPC_Fiji.tif"
            )

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export FLIM TCSPC stack for Fiji/ImageJ",
            default_path,
            "TIFF files (*.tif *.tiff)"
        )

        if not filename:
            return

        # Ensure TIFF suffix
        base, extension = os.path.splitext(
            filename
        )

        if extension.lower() not in [
            ".tif",
            ".tiff"
        ]:

            filename = (
                filename
                + ".tif"
            )

        # ----------------------------------------------------
        # Original array:
        #
        #   Y x X x T
        # ----------------------------------------------------

        flim_yxt = np.asarray(
            self.img["histogram"]
        )

        expected_yxt_shape = (
            self.img["height"],
            self.img["width"],
            self.img["nbins"],
        )

        if (
            flim_yxt.shape
            != expected_yxt_shape
        ):

            self.file_label.setText(
                "FLIM export error: "
                f"histogram shape is "
                f"{flim_yxt.shape}, "
                f"expected "
                f"{expected_yxt_shape}"
            )

            return

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Move the microtime dimension from LAST to FIRST.
        #
        # Before:
        #
        #       Y x X x T
        #
        # After:
        #
        #       T x Y x X
        #
        # Each stack plane is now a normal spatial image.
        # ----------------------------------------------------

        flim_tyx = np.moveaxis(
            flim_yxt,
            2,
            0
        )

        flim_tyx = np.ascontiguousarray(
            flim_tyx
        )

        expected_tyx_shape = (
            self.img["nbins"],
            self.img["height"],
            self.img["width"],
        )

        if (
            flim_tyx.shape
            != expected_tyx_shape
        ):

            self.file_label.setText(
                "FLIM export error: "
                f"converted stack shape "
                f"is {flim_tyx.shape}, "
                f"expected "
                f"{expected_tyx_shape}"
            )

            return

        # ----------------------------------------------------
        # Sanity check:
        #
        # Sum all microtime planes.
        # This should reproduce the intensity image exactly.
        # ----------------------------------------------------

        intensity_from_stack = (
            flim_tyx.sum(
                axis=0,
                dtype=np.uint64
            )
        )

        intensity_reference = np.asarray(
            self.img["intensity"],
            dtype=np.uint64
        )

        intensity_matches = np.array_equal(
            intensity_from_stack,
            intensity_reference
        )

        if not intensity_matches:

            self.file_label.setText(
                "FLIM export stopped: "
                "sum over microtime does not match "
                "the HORIBA intensity image."
            )

            return

        # ----------------------------------------------------
        # Microtime calibration
        #
        # ImageJ treats the leading T dimension as frames.
        # finterval therefore represents the interval between
        # successive TCSPC bins.
        # ----------------------------------------------------

        time_per_bin_s = float(
            self.img["time_per_bin"]
        )

        time_per_bin_ps = (
            time_per_bin_s
            * 1e12
        )

        # ----------------------------------------------------
        # Export as IMAGEJ HYPERSTACK
        #
        # axes = TYX
        #
        # This is the key difference compared with the
        # previous YXT export.
        # ----------------------------------------------------

        try:

            tifffile.imwrite(
                filename,
                flim_tyx,
                imagej=True,
                photometric="minisblack",
                metadata={
                    "axes": "TYX",
                    "finterval": time_per_bin_s,
                }
            )

        except Exception as exc:

            self.file_label.setText(
                f"FLIM stack export error: "
                f"{exc}"
            )

            return

        # ----------------------------------------------------
        # Status information
        # ----------------------------------------------------

        size_mb = (
            flim_tyx.nbytes
            / (1024 ** 2)
        )

        self.file_label.setText(
            f"Exported Fiji FLIM stack: "
            f"{os.path.basename(filename)}"
            f"    |    "
            f"shape T,Y,X = "
            f"{flim_tyx.shape}"
            f"    |    "
            f"{self.img['nbins']} microtime bins"
            f"    |    "
            f"{time_per_bin_ps:.3f} ps/bin"
            f"    |    "
            f"{size_mb:.1f} MB raw"
            f"    |    "
            f"sum(T) = intensity: yes"
        )


    # ========================================================
    # RESET VIEW
    # ========================================================

    def reset_view(self):

        if self.img is None:
            return

        self.set_button_without_signal(
            self.image_roi_button,
            False
        )

        self.set_button_without_signal(
            self.phasor_roi_button,
            False
        )

        self.disable_lassos()

        # ----------------------------------------------------
        # Clear image ROI
        # ----------------------------------------------------

        self.image_roi_mask = None
        self.image_roi_vertices = None

        for patch in (
            self.image_roi_patches
        ):

            try:

                patch.remove()

            except Exception:

                pass

        self.image_roi_patches = []

        # ----------------------------------------------------
        # Clear phasor ROI
        # ----------------------------------------------------

        self.phasor_roi_mask = None
        self.phasor_roi_vertices = None

        if (
            self.phasor_roi_patch
            is not None
        ):

            try:

                self.phasor_roi_patch.remove()

            except Exception:

                pass

        self.phasor_roi_patch = None

        self.clear_phasor_overlay()

        # ----------------------------------------------------
        # Zoom
        # ----------------------------------------------------

        self.reset_image_zoom()

        # ----------------------------------------------------
        # LUT sliders
        # ----------------------------------------------------

        self.reset_lut_sliders()

        self.current_intensity_threshold = (
            self.intensity_data_min
        )

        # ----------------------------------------------------
        # FLIM colourmap
        # ----------------------------------------------------

        self.flim_cmap_combo.blockSignals(
            True
        )

        self.flim_cmap_combo.setCurrentIndex(
            0
        )

        self.flim_cmap_combo.blockSignals(
            False
        )

        self.flim_cmap_name = "viridis"

        cmap = (
            self.get_flim_colormap()
        )

        self.im_lifetime.set_cmap(
            cmap
        )

        self.im_arrival.set_cmap(
            cmap
        )

        # ----------------------------------------------------
        # Phasor filter
        # ----------------------------------------------------

        self.phasor_filter_combo.blockSignals(
            True
        )

        self.phasor_filter_combo.setCurrentIndex(
            0
        )

        self.phasor_filter_combo.blockSignals(
            False
        )

        self.phasor_filter_size = 1

        self.phasor_x_display = (
            self.img["phasor_x"]
            .astype(
                np.float64,
                copy=True
            )
        )

        self.phasor_y_display = (
            self.img["phasor_y"]
            .astype(
                np.float64,
                copy=True
            )
        )

        # ----------------------------------------------------
        # Restore LUT limits
        # ----------------------------------------------------

        self.im_intensity.set_clim(
            vmin=self.intensity_data_min,
            vmax=self.intensity_data_max
        )

        self.im_lifetime.set_clim(
            vmin=self.lifetime_data_min,
            vmax=self.lifetime_data_max
        )

        self.im_arrival.set_clim(
            vmin=self.lifetime_data_min,
            vmax=self.lifetime_data_max
        )

        self.apply_intensity_threshold_to_images()

        if self.cb_intensity is not None:

            self.cb_intensity.update_normal(
                self.im_intensity
            )

        if self.cb_lifetime is not None:

            self.cb_lifetime.update_normal(
                self.im_lifetime
            )

        if self.cb_arrival is not None:

            self.cb_arrival.update_normal(
                self.im_arrival
            )

        self.update_lut_labels()

        # ----------------------------------------------------
        # Bottom plots
        # ----------------------------------------------------

        self.reset_roi_tcspc_axis()

        self.draw_overall_phasor()

        self.reset_roi_phasor_axis()

        self.canvas.draw_idle()


    # ========================================================
    # CLEAR FILE
    # ========================================================

    def clear_file(self):

        self.set_button_without_signal(
            self.image_roi_button,
            False
        )

        self.set_button_without_signal(
            self.phasor_roi_button,
            False
        )

        self.disable_lassos()

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        self.image_roi_mask = None
        self.image_roi_vertices = None
        self.image_roi_patches = []

        self.phasor_roi_mask = None
        self.phasor_roi_vertices = None
        self.phasor_roi_patch = None

        self.phasor_overlay_artists = []

        # ----------------------------------------------------
        # Phasor
        # ----------------------------------------------------

        self.phasor_x_display = None
        self.phasor_y_display = None

        self.phasor_filter_size = 1

        self.phasor_filter_combo.blockSignals(
            True
        )

        self.phasor_filter_combo.setCurrentIndex(
            0
        )

        self.phasor_filter_combo.blockSignals(
            False
        )

        # ----------------------------------------------------
        # FLIM LUT
        # ----------------------------------------------------

        self.flim_cmap_combo.blockSignals(
            True
        )

        self.flim_cmap_combo.setCurrentIndex(
            0
        )

        self.flim_cmap_combo.blockSignals(
            False
        )

        self.flim_cmap_name = "viridis"

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        self.img = None

        self.file_path = None
        self.file_name = None

        self.im_intensity = None
        self.im_lifetime = None
        self.im_arrival = None

        self.cb_intensity = None
        self.cb_lifetime = None
        self.cb_arrival = None

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        self.reset_lut_sliders()

        self.intensity_lut_group.setEnabled(
            False
        )

        self.lifetime_lut_group.setEnabled(
            False
        )

        self.phasor_filter_group.setEnabled(
            False
        )

        self.export_stack_button.setEnabled(
            False
        )

        self.intensity_vmin_label.setText(
            "Vmin / threshold: 0%"
        )

        self.intensity_vmax_label.setText(
            "Vmax: 100%"
        )

        self.lifetime_vmin_label.setText(
            "Vmin: 0%"
        )

        self.lifetime_vmax_label.setText(
            "Vmax: 100%"
        )

        gc.collect()

        self.file_label.setText(
            "No file loaded"
        )

        self.show_empty_figure()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    viewer = HoribaViewer()

    viewer.show()

    sys.exit(
        app.exec_()
    )