import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc_context
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import MEDIUM_SIZE, SMALL_SIZE, savefig, set_style

set_style()

__all__ = ["plot", "plot_log_image"]


def normalize_images_shape(images, subtitles=None):
    if isinstance(images, list):
        images = np.array(images)
        if subtitles is not None:
            subtitles = np.array(subtitles)

    if images.ndim == 2:
        images = images[np.newaxis, np.newaxis]
        if subtitles is not None:
            subtitles = subtitles[np.newaxis, np.newaxis]
    elif images.ndim == 3:
        images = images[np.newaxis]
        if subtitles is not None:
            subtitles = subtitles[np.newaxis]

    return images, subtitles


def get_vmin_vmax(images, same_scale, scale_ref=None, axis="row"):
    """
    Compute shared (vmin, vmax) ranges.

    axis="row": one range per column, computed over the rows in
                ``same_scale`` (or only over row ``scale_ref`` if given).
    axis="col": one range per row, computed over the columns in
                ``same_scale`` (or only over column ``scale_ref`` if given).
    """
    if axis == "col":
        # (rows, cols, ...) -> (cols, rows, ...), so the same logic applies
        images = np.swapaxes(images, 0, 1)
    elif axis != "row":
        raise ValueError("axis must be 'row' or 'col'")

    n = images.shape[1]
    vmin = np.full(n, np.inf)
    vmax = np.full(n, -np.inf)
    refs = [scale_ref] if scale_ref is not None else same_scale

    for i in range(n):
        for ref in refs:
            img = images[ref, i]
            vmin[i] = min(vmin[i], np.nanmin(img))
            vmax[i] = max(vmax[i], np.nanmax(img))
    return vmin, vmax


def plot(
    images,
    titles=None,
    max_imgs=8,
    cmap="inferno",
    cbar=False,
    caption=None,
    same_scale=0,
    fname=None,
    text=None,
    text_row=None,
    scale_row=None,
    suptitle=None,
    swap=False,
    remove_bg=False,
    subtitles=None,
    size_fac=1,
    scale_ranges=None,  # Optional: list of (vmin, vmax) per column
    return_scales=False,
    same_scale_cols=None,
    scale_col=None,
    col_scale_ranges=None,  # Optional: list of (vmin, vmax) per row
):
    """
    Plot images in a grid with optional shared color scaling.

    Parameters
    ----------
    same_scale : int or list
        Rows over which to share the scale (one range per column).
        An int disables row scaling.
    scale_row : int or None
        If given, row-scaled cells take their range from this row only.
    scale_ranges : list of (vmin, vmax) or None
        Optional predefined range per column for row scaling.
    same_scale_cols : list or None
        Columns over which to share the scale (one range per row).
    scale_col : int or None
        If given, column-scaled cells take their range from this column only.
    col_scale_ranges : list of (vmin, vmax) or None
        Optional predefined range per row for column scaling.

    Notes
    -----
    Row and column indices refer to the grid after ``swap`` is applied.
    If a cell is covered by both row and column scaling, row scaling
    takes precedence.

    Returns
    -------
    used_scales : list
        (vmin, vmax) per column from row scaling (None where unused).
        If ``same_scale_cols`` is given, returns
        ``(used_scales, used_col_scales)``, where ``used_col_scales`` holds
        (vmin, vmax) per row from column scaling.
    """
    images, subtitles = normalize_images_shape(images, subtitles)

    if swap:
        images = np.swapaxes(images, 0, 1)
        if subtitles is not None:
            subtitles = np.swapaxes(subtitles, 0, 1)

    rows = min(max_imgs, images.shape[0])
    cols = images.shape[1]

    if isinstance(titles, str):
        titles = [titles]
    if titles and len(titles) != cols:
        raise ValueError("Title list should match number of columns")

    # Row scaling: one (vmin, vmax) per column
    used_scales = [None] * cols
    if scale_ranges is not None:
        used_scales = scale_ranges
    elif not isinstance(same_scale, int):
        vmin, vmax = get_vmin_vmax(images, same_scale, scale_row, axis="row")
        used_scales = [(vmin[c], vmax[c]) for c in range(cols)]

    # Column scaling: one (vmin, vmax) per row
    used_col_scales = [None] * images.shape[0]
    if col_scale_ranges is not None:
        used_col_scales = col_scale_ranges
    elif same_scale_cols is not None:
        vmin, vmax = get_vmin_vmax(images, same_scale_cols, scale_col, axis="col")
        used_col_scales = [(vmin[r], vmax[r]) for r in range(images.shape[0])]

    with rc_context(
        rc={
            "axes.labelpad": 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        figsize = (
            (2.5 * cols * size_fac, rows * 2.5 * size_fac)
            if cbar
            else (2 * cols * size_fac, rows * 2 * size_fac)
        )
        fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=figsize)
        axs = np.atleast_2d(axs)

        for r in range(rows):
            for c in range(cols):
                ax = axs[r, c]

                row_scaled = not isinstance(same_scale, int) and r in same_scale
                col_scaled = same_scale_cols is not None and c in same_scale_cols

                if row_scaled:
                    scale = used_scales[c]
                elif col_scaled:
                    scale = used_col_scales[r]
                else:
                    scale = None

                if scale is not None:
                    im = ax.imshow(
                        images[r, c], cmap=cmap, vmin=scale[0], vmax=scale[1]
                    )
                else:
                    im = ax.imshow(images[r, c], cmap=cmap)

                if titles and r == 0:
                    ax.set_title(titles[c], size=MEDIUM_SIZE)
                if subtitles is not None:
                    ax.set_title(subtitles[r, c], size=MEDIUM_SIZE)

                if cbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
                    fmt.set_powerlimits((-3, 2))
                    colbar = fig.colorbar(
                        im, cax=cax, orientation="vertical", format=fmt
                    )
                    colbar.ax.tick_params(labelsize=SMALL_SIZE)
                    colbar.ax.yaxis.get_offset_text().set(size=SMALL_SIZE)

        if text is not None and text_row is not None:
            for ax, txt in zip(axs[text_row], text):
                ax.text(
                    0.65,
                    0.8,
                    txt,
                    size=MEDIUM_SIZE,
                    color="white",
                    transform=ax.transAxes,
                )

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        if caption:
            for cap, ax in zip(caption, axs[:, 0]):
                ax.set_ylabel(cap, fontsize=MEDIUM_SIZE)

        if suptitle:
            fig.suptitle(suptitle)

        savefig(fname, remove_bg=remove_bg)

    if return_scales:
        if same_scale_cols is not None:
            return used_scales, used_col_scales
        return used_scales


# %%
def plot_log_image(im, figsize=(8, 8), cmap="inferno", fname=None, remove_bg=False):
    im = np.clip(im, 0, None)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Display the image in log scale but preserve actual flux scaling
    im_disp = ax.imshow(
        im + 1e-9,
        norm=LogNorm(vmin=max(im.min(), 1e-9), vmax=im.max()),
        cmap=cmap,
        origin="lower",
    )

    # Remove all axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # --- Custom colorbar on the right ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 2))

    colbar = fig.colorbar(im_disp, cax=cax, orientation="vertical", format=fmt)
    colbar.set_label("Flux [Jy]", fontsize=MEDIUM_SIZE, labelpad=10)
    colbar.ax.tick_params(labelsize=SMALL_SIZE)
    colbar.ax.yaxis.get_offset_text().set(size=SMALL_SIZE)

    if fname:
        savefig(fname, remove_bg=remove_bg)
    else:
        plt.show()
