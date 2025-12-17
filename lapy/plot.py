"""Functions for plotting Tet and Tria Meshes with overlays.

For visualizing results in a juypter notebook use this::

    .. code-block:: python

        import plotly
        plotly.offline.init_notebook_mode(connected=True)
        ...
"""

import re
from bisect import bisect

import numpy as np
import plotly
import plotly.graph_objs as go
from matplotlib.colors import LinearSegmentedColormap

from . import TetMesh


def _get_color_levels():
    """Return a pre-set colorscale.

    Returns
    -------
    colorscale: array_like of shape (38, 2)
        Vector color for different levels.
    """
    color1 = "rgb(55, 155, 255)"
    color2 = "rgb(255, 255, 0)"
    colorscale = [
        [0, color1],
        [0.09999, color1],
        [0.1, color2],
        [0.105, color2],
        [0.1050001, color1],
        [0.19999, color1],
        [0.2, color2],
        [0.205, color2],
        [0.2050001, color1],
        [0.29999, color1],
        [0.3, color2],
        [0.305, color2],
        [0.3050001, color1],
        [0.399999, color1],
        [0.4, color2],
        [0.405, color2],
        [0.4050001, color1],
        [0.49999, color1],
        [0.5, color2],
        [0.505, color2],
        [0.5050001, color1],
        [0.599999, color1],
        [0.6, color2],
        [0.605, color2],
        [0.6050001, color1],
        [0.69999, color1],
        [0.7, color2],
        [0.705, color2],
        [0.7050001, color1],
        [0.799999, color1],
        [0.8, color2],
        [0.805, color2],
        [0.8050001, color1],
        [0.89999, color1],
        [0.9, color2],
        [0.905, color2],
        [0.9050001, color1],
        [1, color1],
    ]
    return colorscale


def _get_colorscale(vmin, vmax):
    """Put together a colorscale map depending on the range of v-values.

    Parameters
    ----------
    vmin : float
        Minimum value.
    vmax : float
        Maximum value.

    Returns
    -------
    colorscale: array_like of shape (2,2)
        Colorscale map.
    """
    if vmin > vmax:
        raise ValueError("incorrect relation between vmin and vmax")
    # color definitions
    posstart = "rgb(255, 0, 0)"
    posstop = "rgb(255, 255, 51)"
    negstart = "rgb(51, 255, 255)"
    negstop = "rgb(0, 0, 255)"
    zcolor = "rgb(190, 210, 220)"
    if vmin > 0:
        # only positive values
        colorscale = [[0, posstart], [1, posstop]]
    elif vmax < 0:
        # only negative values
        colorscale = [[0, negstart], [1, negstop]]
    else:
        # both pos and negative (here extra color for values around zero)
        zz = -vmin / (vmax - vmin)
        eps = 0.000000001
        zero = 0.001
        if zz < (eps + zero):
            # only very few negative values (map to zero color)
            colorscale = [
                [0, zcolor],
                [zero, zcolor],
                [zero + eps, posstart],
                [1, posstop],
            ]
        elif zz > (1.0 - eps - zero):
            # only very few positive values (map to zero color)
            colorscale = [
                [0, negstart],
                [1 - zero - eps, negstop],
                [1 - zero, zcolor],
                [1, zcolor],
            ]
        else:
            # sufficient negative and positive values
            colorscale = [
                [0, negstart],
                [zz - zero - eps, negstop],
                [zz - zero, zcolor],
                [zz + zero, zcolor],
                [zz + zero + eps, posstart],
                [1, posstop],
            ]
    return colorscale


def _get_colorval(t, colormap):
    """Turn a scalar value into a color value.

    Parameters
    ----------
    t : float
        Scalar must be 0...1.
    colormap : array_like
        List of values and color code strings (with entries at least for 0 and 1).

    Returns
    -------
    cstr/*: str
        Interpolated color for this value of t.
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be between 0 and 1")
    if t == 0:
        return colormap[0][1]
    if t == 1:
        return colormap[-1][1]
    # ok here we need to interpolate
    # first find two colors before and after
    columns = list(zip(*colormap))
    pos = bisect(columns[0], t)
    # compute param between pos-1 and pos values
    if len(columns[0]) < pos + 1 or pos == 0:
        #print(f"pos: {pos}")
        #print(f"t: {t}")
        #print(columns[0])
        raise ValueError("t not in range?")
    tt = (t - columns[0][pos - 1]) / (columns[0][pos] - columns[0][pos - 1])
    # get color before and after as array of 3 ints
    rv1 = np.array(list(map(int, re.findall("[0-9]+", columns[1][pos - 1]))))
    rv2 = np.array(list(map(int, re.findall("[0-9]+", columns[1][pos]))))
    # compute new color via linear interpolation
    cval = np.rint(rv1 + tt * (rv2 - rv1)).astype(int)
    # format as string again
    cstr = f"rgb({cval[0]:d}, {cval[1]:d}, {cval[2]:d})"
    return cstr


def _map_z2color(zval, colormap, zmin, zmax):
    """Map the normalized value zval to a corresponding color in the colormap.

    Parameters
    ----------
    zval : float
        Value to be mapped.
    colormap : matplotlib.colors.LinearSegmentedColormap | array
        List of values and color code strings.
    zmin : float
        Minimum.
    zmax : float
        Maximum.

    Returns
    -------
    rgb : str
        Corresponding color of the zval.
    """
    if zmin > zmax:
        raise ValueError("incorrect relation between zmin and zmax")

    t = (zval - zmin) / float(zmax - zmin)  # normalize val
    if isinstance(colormap, LinearSegmentedColormap):
        r, g, b, alpha = colormap(t)
        rgb = (
            "rgb("
            + f"{int(r * 255 + 0.5):d}"
            + ","
            + f"{int(g * 255 + 0.5):d}"
            + ","
            + f"{int(b * 255 + 0.5):d}"
            + ")"
        )
    else:
        rgb = _get_colorval(t, colormap)

    return rgb


def plot_tet_mesh(
    tetra,
    vfunc=None,
    plot_edges=False,
    plot_levels=False,
    tfunc=None,
    cutting=None,
    edge_color="rgb(50,50,50)",
    html_output=False,
    width=800,
    height=800,
    flatshading=False,
    xrange=None,
    yrange=None,
    zrange=None,
    showcaxis=False,
    caxis=None,
):
    """Plot tetra meshes.

    The tetra mesh will be converted to its tria boundary mesh,
    and only this will be plotted.

    Parameters
    ----------
    tetra : lapy.TetMesh
        Tetraheral mesh to plot.
    vfunc : array_like, Default=None
        Scalar function at vertices.
    plot_edges : bool, Default=False
        Whether to plot edges or not.
    plot_levels : bool, Default=False
        Whether to plot levels or not.
    tfunc : array_like, Default=None
        3d vector function of gradient.
    cutting : str, Default=None
        To view the 'interior' of the tetra mesh, one or more cutting
        criteria can be defined as input arguments to this function:
        e.g. cutting=('x<-10') or cutting=('z>=5') or cutting=('f>4')
        where x,y,z represent dimensions 0,1,2 of the vertex array,
        and f represents the vfunc (which cannot be None if f is used
        to define a cutting criterion).
    edge_color : str, Default="rgb(50,50,50)"
        Color of the edge.
    html_output : bool, Default=False
        Whether or not to give out as html output.
    width : int, Default=800
        Width of the plot (in px).
    height : int, Default=800
        Height  of the plot (in px).
    flatshading : bool, Default=False
        Whether normal smoothing is applied to the meshes or not.
    xrange : list or tuple of shape (2, 1)
        Sets the range of the x-axis.
    yrange : list or tuple of shape (2, 1)
        Sets the range of the y-axis.
    zrange : list or tuple of shape (2, 1)
        Sets the range of the z-axis.
    showcaxis : bool, Default=False
        Whether a colorbar is displayed or not.
    caxis : list or tuple of shape (2, 1):
        Sets the bound of the color domain.
        caxis[0] is lower bound caxis[1] upper bound.
        Elements are int or float.
    """
    if type(tetra).__name__ != "TetMesh":
        raise ValueError("plot_tet_mesh works only on TetMesh class")

    # evaluate cutting criteria
    if cutting is not None:
        # check inputs
        if not isinstance(cutting, list):
            cutting = [cutting]

        # check if vfunc is defined when functional thresholds are used, otherwise exit
        if any(["f" in x for x in cutting]) and vfunc is None:
            raise ValueError(
                "Need to specify vfunc if 'f' is used"
                " within the 'cutting' argument, exiting."
            )

        # create criteria from cutting info
        criteria = (
            "("
            + ")&(".join(
                [
                    x.replace("x", "tetra.v[:,0]")
                    .replace("y", "tetra.v[:,1]")
                    .replace("z", "tetra.v[:,2]")
                    .replace("f", "vfunc")
                    for x in cutting
                ]
            )
            + ")"
        )

        # apply criteria to find matching vertices
        sel = np.where(eval(criteria))

        # find matching tetras
        tidx = np.where(np.sum(np.isin(tetra.t, sel), axis=1) == 4)[0]
        tsel = tetra.t[tidx, :]
        tfunc_sel = None
        if tfunc is not None:
            tfunc_sel = tfunc[tidx]

    else:
        # select all tetras
        tsel = tetra.t
        tfunc_sel = None
        if tfunc is not None:
            tfunc_sel = tfunc

    subtetra = TetMesh(tetra.v, tsel)
    # ensure mesh is oriented
    subtetra.orient_()

    # convert to tria mesh (=get boundary)
    if tfunc_sel is not None:
        tria_bnd, tfunc_tria = subtetra.boundary_tria(tfunc_sel)
    else:
        tfunc_tria = None
        tria_bnd = subtetra.boundary_tria()

    # check if tria mesh is oriented
    if not tria_bnd.is_oriented():
        tria_bnd.orient_()  # should not get here!

    # run plot_tria_mesh on boundary tria
    plot_tria_mesh(
        tria_bnd,
        vfunc=vfunc,
        plot_edges=plot_edges,
        plot_levels=plot_levels,
        tfunc=tfunc_tria,
        edge_color=edge_color,
        html_output=html_output,
        width=width,
        height=height,
        flatshading=flatshading,
        xrange=xrange,
        yrange=yrange,
        zrange=zrange,
        showcaxis=showcaxis,
        caxis=caxis,
    )


def plot_tria_mesh(
    tria,
    vfunc=None,
    tfunc=None,
    vcolor=None,
    tcolor=None,
    showcaxis=False,
    caxis=None,
    xrange=None,
    yrange=None,
    zrange=None,
    plot_edges=False,
    plot_levels=False,
    edge_color="rgb(50,50,50)",
    tic_color="rgb(50,200,10)",
    background_color=None,
    flatshading=False,
    width=800,
    height=800,
    camera=None,
    html_output=False,
    export_png=None,
    scale_png=1.0,
    no_display=False,
):
    """Plot tria mesh.

    Parameters
    ----------
    tria : lapy.TriaMesh
        Triangle mesh to plot.
    vfunc : array_like, Default=None
        Scalar function at vertices.
    tfunc : array_like, Default=None
        3d vector function of gradient.
    vcolor : list of str, Default=None
        Sets the color of each vertex.
    tcolor : list of str, Default=None
         Sets the color of each face.
    showcaxis : bool, Default=False
        Whether a colorbar is displayed or not.
    caxis : list or tuple of shape (2, 1):
        Sets the bound of the color domain.
        caxis[0] is lower bound caxis[1] upper bound.
        Elements are int or float.
    xrange : list or tuple of shape (2, 1)
        Sets the range of the x-axis.
    yrange : list or tuple of shape (2, 1)
        Sets the range of the y-axis.
    zrange : list or tuple of shape (2, 1)
        Sets the range of the z-axis.
    plot_edges : bool, Default=False
        Whether to plot edges or not.
    plot_levels : bool, Default=False
        Whether to plot levels or not.
    edge_color : str, Default="rgb(50,50,50)"
        Color of the edges.
    tic_color : str, Default="rgb(50,200,10)"
        Color of the ticks.
    background_color : str, Default=None
        Color of background.
    flatshading : bool, Default=False
        Whether normal smoothing is applied to the meshes or not.
    width : int, Default=800
        Width of the plot (in px).
    height : int, Default=800
        Height  of the plot (in px).
    camera : dict of str, Default=None
        Camera describing center, eye and up direction.
    html_output : bool, Default=False
        Whether or not to give out as html output.
    export_png : str, Default=None
        Local file path or file object to write the image to.
    scale_png : int or float
        Scale factor of image. >1.0 increase resolution; <1.0 decrease resolution.
    no_display : bool, Default=False
        Whether to plot on display or not.
    """
    # interesting example codes:
    # https://plot.ly/~empet/14749/mesh3d-with-intensities-and-flatshading/#/

    if type(tria).__name__ != "TriaMesh":
        raise ValueError("plot_tria_mesh works only on TriaMesh class")

    if (vfunc is not None or tfunc is not None) and (
        vcolor is not None or tcolor is not None
    ):
        raise ValueError(
            "plot_tria_mesh can only use either vfunc/tfunc or vcolor/tcolor,"
            " but not both at the same time"
        )

    if vcolor is not None and tcolor is not None:
        raise ValueError(
            "plot_tria_mesh can only use either vcolor or tcolor,"
            " but not both at the same time"
        )

    x, y, z = zip(*tria.v)
    i, j, k = zip(*tria.t)

    vlines = []
    if vfunc is None:
        if tfunc is None:
            triangles = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                flatshading=flatshading,
                vertexcolor=vcolor,
                facecolor=tcolor,
            )
        elif tfunc.ndim == 1 or (tfunc.ndim == 2 and np.min(tfunc.shape) == 1):
            # scalar tfunc
            min_fcol = np.min(tfunc)
            max_fcol = np.max(tfunc)
            # special treatment for constant functions
            if np.abs(min_fcol - max_fcol) < 0.0001:
                if np.abs(max_fcol) > 0.0001:
                    min_fcol = -np.abs(min_fcol)
                    max_fcol = np.abs(max_fcol)
                else:  # both are zero
                    min_fcol = -1
                    max_fcol = 1
            # if min_fcol >= 0 and max_fcol <= 1:
            #    min_fcol = 0
            #    max_fcol = 1
            # colormap = cm.RdBu
            colormap = _get_colorscale(min_fcol, max_fcol)
            facecolor = [_map_z2color(zz, colormap, min_fcol, max_fcol) for zz in tfunc]
            # for tria colors overwrite flatshading to be true:
            triangles = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                facecolor=facecolor,
                flatshading=True,
            )
        elif tfunc.ndim == 2 and np.min(tfunc.shape) == 3:
            # vector tfunc
            s = 0.7 * tria.avg_edge_length()
            centroids = (1.0 / 3.0) * (
                tria.v[tria.t[:, 0], :]
                + tria.v[tria.t[:, 1], :]
                + tria.v[tria.t[:, 2], :]
            )
            xv = np.column_stack(
                (
                    centroids[:, 0],
                    centroids[:, 0] + s * tfunc[:, 0],
                    np.full(tria.t.shape[0], np.nan),
                )
            ).reshape(-1)
            yv = np.column_stack(
                (
                    centroids[:, 1],
                    centroids[:, 1] + s * tfunc[:, 1],
                    np.full(tria.t.shape[0], np.nan),
                )
            ).reshape(-1)
            zv = np.column_stack(
                (
                    centroids[:, 2],
                    centroids[:, 2] + s * tfunc[:, 2],
                    np.full(tria.t.shape[0], np.nan),
                )
            ).reshape(-1)
            vlines = go.Scatter3d(
                x=xv,
                y=yv,
                z=zv,
                mode="lines",
                line=dict(color=tic_color, width=2),
            )
            triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, flatshading=flatshading)
        else:
            raise ValueError(
                "tfunc should be scalar (face color) or 3d for each triangle"
            )

    elif vfunc.ndim == 1 or (vfunc.ndim == 2 and np.min(vfunc.shape) == 1):
        # scalar vfunc
        if plot_levels:
            colorscale = _get_color_levels()
        else:
            colorscale = _get_colorscale(min(vfunc), max(vfunc))

        triangles = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            intensity=vfunc,
            colorscale=colorscale,
            flatshading=flatshading,
        )
    elif vfunc.ndim == 2 and np.min(vfunc.shape) == 3:
        # vector vfunc
        s = 0.7 * tria.avg_edge_length()
        xv = np.column_stack(
            (
                tria.v[:, 0],
                tria.v[:, 0] + s * vfunc[:, 0],
                np.full(tria.v.shape[0], np.nan),
            )
        ).reshape(-1)
        yv = np.column_stack(
            (
                tria.v[:, 1],
                tria.v[:, 1] + s * vfunc[:, 1],
                np.full(tria.v.shape[0], np.nan),
            )
        ).reshape(-1)
        zv = np.column_stack(
            (
                tria.v[:, 2],
                tria.v[:, 2] + s * vfunc[:, 2],
                np.full(tria.v.shape[0], np.nan),
            )
        ).reshape(-1)
        vlines = go.Scatter3d(
            x=xv, y=yv, z=zv, mode="lines", line=dict(color=tic_color, width=2)
        )
        triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, flatshading=flatshading)
    else:
        raise ValueError("vfunc should be scalar or 3d for each vertex")

    if plot_edges:
        # 4 points = three edges for each tria, nan to separate triangles
        # this plots every edge twice (except boundary edges)
        xe = np.column_stack(
            (
                tria.v[tria.t[:, 0], 0],
                tria.v[tria.t[:, 1], 0],
                tria.v[tria.t[:, 2], 0],
                tria.v[tria.t[:, 0], 0],
                np.full(tria.t.shape[0], np.nan),
            )
        ).reshape(-1)
        ye = np.column_stack(
            (
                tria.v[tria.t[:, 0], 1],
                tria.v[tria.t[:, 1], 1],
                tria.v[tria.t[:, 2], 1],
                tria.v[tria.t[:, 0], 1],
                np.full(tria.t.shape[0], np.nan),
            )
        ).reshape(-1)
        ze = np.column_stack(
            (
                tria.v[tria.t[:, 0], 2],
                tria.v[tria.t[:, 1], 2],
                tria.v[tria.t[:, 2], 2],
                tria.v[tria.t[:, 0], 2],
                np.full(tria.t.shape[0], np.nan),
            )
        ).reshape(-1)

        # define the lines to be plotted
        lines = go.Scatter3d(
            x=xe,
            y=ye,
            z=ze,
            mode="lines",
            line=dict(color=edge_color, width=1.5),
        )

        data = [triangles, lines]

    else:
        data = [triangles]

    if vlines:
        data.append(vlines)

    # line_marker = dict(color='#0066FF', width=2)

    noaxis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis),
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
    )

    if camera is not None:
        layout.scene.camera.center.update(camera["center"])
        layout.scene.camera.eye.update(camera["eye"])
        layout.scene.camera.up.update(camera["up"])

    if xrange is not None:
        layout.scene.xaxis.update(range=xrange)
    if yrange is not None:
        layout.scene.yaxis.update(range=yrange)
    if zrange is not None:
        layout.scene.zaxis.update(range=zrange)

    data[0].update(showscale=showcaxis)

    if caxis is not None:
        data[0].update(cmin=caxis[0])
        data[0].update(cmax=caxis[1])

    fig = go.Figure(data=data, layout=layout)

    if no_display is False:
        if not html_output:
            plotly.offline.iplot(fig)
        else:
            plotly.offline.plot(fig)

    if export_png is not None:
        fig.write_image(export_png, scale=scale_png)
