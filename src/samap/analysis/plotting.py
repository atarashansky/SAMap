"""Plotting functions for SAMap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def _q(x: Any) -> NDArray[Any]:
    """Convert input to numpy array."""
    return np.array(list(x))


def sankey_plot(
    M: pd.DataFrame,
    species_order: list[str] | None = None,
    align_thr: float = 0.1,
    **params: Any,
) -> Any:
    """Generate a sankey plot of cell type mappings.

    Parameters
    ----------
    M : pd.DataFrame
        Mapping table from `get_mapping_scores` (second output).
    species_order : list, optional
        Order of species left-to-right, e.g., ['hu', 'le', 'ms'].
    align_thr : float, optional
        Alignment score threshold. Default 0.1.
    **params
        Keyword arguments for customization:
        - node_padding: int, padding between nodes (default 20)
        - node_thickness: int, thickness of nodes (default 20)
        - node_opacity: float, opacity of nodes (default 1.0)
        - link_opacity: float, opacity of links (default 0.5)
        - height: int, figure height in pixels (default 800)
        - width: int, figure width in pixels (default 1000)
        - bgcolor: str, background color (default "white")
        - font_size: int, label font size (default 12)
        - font_color: str, label font color (default "black")
        - title: str, plot title (default None)
        - colorscale: str, colorscale for nodes (default "Viridis")

    Returns
    -------
    plotly.graph_objects.Figure
        Sankey plot figure.

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        raise ImportError(
            "Please install plotly with `pip install plotly`. "
            "For Jupyter support, also run: pip install jupyterlab ipywidgets"
        ) from None

    # Auto-detect notebook environment and set renderer
    if pio.renderers.default == "plotly_mimetype":
        try:
            # Check if we're in a Jupyter environment
            get_ipython()  # type: ignore[name-defined]
            pio.renderers.default = "notebook"
        except NameError:
            pass  # Not in IPython/Jupyter

    if species_order is not None:
        ids = np.array(species_order)
    else:
        ids = np.unique([x.split("_")[0] for x in M.index])

    # Build source/target/value data from mapping matrix
    if len(ids) > 2:
        d = M.values.copy()
        d[d < align_thr] = 0
        x, y = d.nonzero()
        x, y = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0).T
        values = d[x, y]
        nodes = _q(M.index)

        node_pairs = nodes[np.vstack((x, y)).T]
        sn1 = _q([xi.split("_")[0] for xi in node_pairs[:, 0]])
        sn2 = _q([xi.split("_")[0] for xi in node_pairs[:, 1]])
        # Filter to only adjacent species pairs in the ordering
        filt = np.logical_or(
            np.logical_or(
                np.logical_and(sn1 == ids[0], sn2 == ids[1]),
                np.logical_and(sn1 == ids[1], sn2 == ids[0]),
            ),
            np.logical_or(
                np.logical_and(sn1 == ids[1], sn2 == ids[2]),
                np.logical_and(sn1 == ids[2], sn2 == ids[1]),
            ),
        )
        x, y, values = x[filt], y[filt], values[filt]

        # Create depth mapping for node ordering
        depth_dict = dict(zip(ids, list(np.arange(len(ids)))))
        data = nodes[np.vstack((x, y))].T
        # Ensure source comes before target in species order
        for i in range(data.shape[0]):
            if depth_dict[data[i, 0].split("_")[0]] > depth_dict[data[i, 1].split("_")[0]]:
                data[i, :] = data[i, ::-1]
        R = pd.DataFrame(data=data, columns=["source", "target"])
        R["value"] = values
    else:
        d = M.values.copy()
        d[d < align_thr] = 0
        x, y = d.nonzero()
        x, y = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0).T
        values = d[x, y]
        nodes = _q(M.index)
        R = pd.DataFrame(data=nodes[np.vstack((x, y))].T, columns=["source", "target"])
        R["value"] = values
        depth_dict = None

    # Build node list from unique sources and targets
    all_nodes = pd.concat([R["source"], R["target"]]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    # Map source/target names to indices
    source_indices = [node_to_idx[src] for src in R["source"]]
    target_indices = [node_to_idx[tgt] for tgt in R["target"]]

    # Assign colors based on species
    species_list = [node.split("_")[0] for node in all_nodes]
    unique_species = list(dict.fromkeys(species_list))  # preserve order
    colorscale = params.get("colorscale", "Viridis")

    # Generate colors for each species
    n_species = len(unique_species)
    if n_species <= 10:
        # Use qualitative colors for small number of species
        default_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        species_colors = {sp: default_colors[i % len(default_colors)]
                         for i, sp in enumerate(unique_species)}
    else:
        # Fall back to colorscale
        import colorsys
        species_colors = {}
        for i, sp in enumerate(unique_species):
            hue = i / n_species
            rgb = colorsys.hls_to_rgb(hue, 0.5, 0.7)
            species_colors[sp] = f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})"

    node_colors = [species_colors[sp] for sp in species_list]

    # Create link colors (lighter versions of source node colors)
    link_opacity = params.get("link_opacity", 0.5)
    link_colors = []
    for src in R["source"]:
        sp = src.split("_")[0]
        base_color = species_colors[sp]
        # Add opacity to the color
        if base_color.startswith("#"):
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16)
            b = int(base_color[5:7], 16)
            link_colors.append(f"rgba({r},{g},{b},{link_opacity})")
        else:
            link_colors.append(base_color.replace("rgb", "rgba").replace(")", f",{link_opacity})"))

    # Extract parameters
    node_padding = params.get("node_padding", 20)
    node_thickness = params.get("node_thickness", 20)
    node_opacity = params.get("node_opacity", 1.0)
    height = params.get("height", 800)
    width = params.get("width", 1000)
    bgcolor = params.get("bgcolor", "white")
    font_size = params.get("font_size", 12)
    font_color = params.get("font_color", "black")
    title = params.get("title", None)

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=node_padding,
            thickness=node_thickness,
            line=dict(color="black", width=0.5),
            label=list(all_nodes),
            color=node_colors,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=list(R["value"]),
            color=link_colors,
        ),
    )])

    fig.update_layout(
        title=dict(text=title) if title else None,
        font=dict(size=font_size, color=font_color),
        paper_bgcolor=bgcolor,
        plot_bgcolor=bgcolor,
        height=height,
        width=width,
    )

    return fig


