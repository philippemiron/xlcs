import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs


def gom_map(ax):
    """
    Create a map of the Gulf of Mexico with clean axis, coastline, and land
    Args:
        ax: matplotlib axis
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
    """Colorbar position and format properly"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02, axes_class=plt.Axes)
    cb = fig.colorbar(var, cax=cax, format=fmt)
    if range_limit:
        cb.mappable.set_clim(range_limit)
    cb.ax.tick_params(which="major", labelsize=6, length=3, width=0.5, pad=0.05)
    return cb
