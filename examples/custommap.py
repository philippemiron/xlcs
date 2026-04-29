"""Cartopy map helpers for Gulf of Mexico example plots."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def gom_map(ax):
    """Configure a cartopy axis as a Gulf of Mexico map.

    Args:
        ax: matplotlib cartopy axis

    """
    ax.add_feature(cfeature.LAND, facecolor="grey", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.1, zorder=1)
    ax.set_xlim([-98, -77])
    ax.set_ylim([18, 31])
    ax.set_xticks([-95, -90, -85, -80], crs=ccrs.PlateCarree())
    ax.set_yticks([20, 25, 30], crs=ccrs.PlateCarree())
    ax.tick_params(axis="both", labelsize=6, pad=1)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())


def add_colorbar(fig, ax, var, fmt=None, range_limit=None):
    """Add a right-side colorbar sized to match the axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02, axes_class=plt.Axes)
    cb = fig.colorbar(var, cax=cax, format=fmt)
    if range_limit:
        cb.mappable.set_clim(range_limit)
    cb.ax.tick_params(which="major", labelsize=6, length=3, width=0.5, pad=0.05)
    return cb
