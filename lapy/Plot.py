"""

Dependency:
    plotly 3.6

In jupyter notebook do this:
    import plotly
    plotly.offline.init_notebook_mode(connected=True)

"""
import numpy as np
import plotly
import plotly.graph_objs as go
import matplotlib.cm as cm
from .TetMesh import TetMesh


def _get_color_levels():
    color1 = 'rgb(55, 155, 255)'
    color2 = 'rgb(255, 255, 0)'
    colorscale = [[0, color1],
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
                  [1, color1]]
    return colorscale


def _map_z2color(zval, colormap, vmin, vmax):
    # map the normalized value zval to a corresponding color in the colormap
    if vmin > vmax:
        raise ValueError('incorrect relation between vmin and vmax')

    t = (zval - vmin) / float((vmax - vmin))  # normalize val
    r, g, b, alpha = colormap(t)

    return 'rgb(' + '{:d}'.format(int(r * 255 + 0.5)) + ',' + '{:d}'.format(int(g * 255 + 0.5)) + ',' + \
           '{:d}'.format(int(b * 255 + 0.5)) + ')'


def plot_tria_mesh(tria, vfunc=None, plot_edges=None, plot_levels=False, tfunc=None, edge_color='rgb(50,50,50)',
                   tic_color='rgb(50,200,10)', html_output=False, width=800, height=800, flatshading=False,
                   xrange=None, yrange=None, zrange=None, showcaxis=False, caxis=None):
    # interesting example codes:
    # https://plot.ly/~empet/14749/mesh3d-with-intensities-and-flatshading/#/
    #
    if type(tria).__name__ is not "TriaMesh":
        raise ValueError('plot_tria_mesh works only on TriaMesh class')

    x, y, z = zip(*tria.v)
    i, j, k = zip(*tria.t)

    vlines = []
    if vfunc is None:
        if tfunc is None:
            triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, flatshading=flatshading)
        elif tfunc.ndim == 1 or (tfunc.ndim == 2 and np.min(tfunc.shape) == 1):
            min_fcol = np.min(tfunc)
            max_fcol = np.max(tfunc)
            if min_fcol >= 0 and max_fcol <= 1:
                min_fcol = 0
                max_fcol = 1
            colormap = cm.RdBu
            facecolor = [_map_z2color(zz, colormap, min_fcol, max_fcol) for zz in tfunc]
            # for tria colors overwrite flatshading to be true:
            triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=facecolor, flatshading=True)
        elif tfunc.ndim == 2 and np.min(tfunc.shape) == 3:
            s = 0.7 * tria.avg_edge_length()
            centroids = (1.0/3.0) * (tria.v[tria.t[:, 0], :] + tria.v[tria.t[:, 1], :] + tria.v[tria.t[:, 2], :])
            xv = np.column_stack(
                (centroids[:, 0], centroids[:, 0] + s * tfunc[:, 0], np.full(tria.t.shape[0], np.nan))).reshape(-1)
            yv = np.column_stack(
                (centroids[:, 1], centroids[:, 1] + s * tfunc[:, 1], np.full(tria.t.shape[0], np.nan))).reshape(-1)
            zv = np.column_stack(
                (centroids[:, 2], centroids[:, 2] + s * tfunc[:, 2], np.full(tria.t.shape[0], np.nan))).reshape(-1)
            vlines = go.Scatter3d(x=xv, y=yv, z=zv, mode='lines', line=dict(color=tic_color, width=2))
            triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, flatshading=flatshading)
        else:
            raise ValueError('tfunc should be scalar (face color) or 3d for each triangle')

    elif vfunc.ndim == 1 or (vfunc.ndim == 2 and np.min(vfunc.shape) == 1):
        if min(vfunc) >= 0 or max(vfunc) <= 0:
            colorscale = [[0, 'rgb(255, 0, 0)'],
                          [1, 'rgb(255, 255, 51)']]
        else:
            zz = -min(vfunc) / (max(vfunc) - min(vfunc))
            colorscale = [[0, 'rgb(51, 255, 255)'],
                          [zz - 0.00000001, 'rgb(0, 0, 255)'],
                          [zz, 'rgb(255, 0, 0)'],
                          [1, 'rgb(255, 255, 51)']]
        if plot_levels:
            colorscale = _get_color_levels()
        triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, intensity=vfunc, colorscale=colorscale,
                              flatshading=flatshading)
    elif vfunc.ndim == 2 and np.min(vfunc.shape) == 3:
        s = 0.7 * tria.avg_edge_length()
        xv = np.column_stack(
            (tria.v[:, 0], tria.v[:, 0] + s * vfunc[:, 0], np.full(tria.v.shape[0], np.nan))).reshape(-1)
        yv = np.column_stack(
            (tria.v[:, 1], tria.v[:, 1] + s * vfunc[:, 1], np.full(tria.v.shape[0], np.nan))).reshape(-1)
        zv = np.column_stack(
            (tria.v[:, 2], tria.v[:, 2] + s * vfunc[:, 2], np.full(tria.v.shape[0], np.nan))).reshape(-1)
        vlines = go.Scatter3d(x=xv, y=yv, z=zv, mode='lines', line=dict(color=tic_color, width=2))
        triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, flatshading=flatshading)
    else:
        raise ValueError('vfunc should be scalar or 3d for each vertex')

    if plot_edges:
        # 4 points = three edges for each tria, nan to separate triangles
        # this plots every edge twice (except boundary edges)
        xe = np.column_stack(
            (tria.v[tria.t[:, 0], 0], tria.v[tria.t[:, 1], 0], tria.v[tria.t[:, 2], 0],
             tria.v[tria.t[:, 0], 0], np.full(tria.t.shape[0], np.nan))).reshape(-1)
        ye = np.column_stack(
            (tria.v[tria.t[:, 0], 1], tria.v[tria.t[:, 1], 1], tria.v[tria.t[:, 2], 1],
             tria.v[tria.t[:, 0], 1], np.full(tria.t.shape[0], np.nan))).reshape(-1)
        ze = np.column_stack(
            (tria.v[tria.t[:, 0], 2], tria.v[tria.t[:, 1], 2], tria.v[tria.t[:, 2], 2],
             tria.v[tria.t[:, 0], 2], np.full(tria.t.shape[0], np.nan))).reshape(-1)

        # define the lines to be plotted
        lines = go.Scatter3d(x=xe,
                             y=ye,
                             z=ze,
                             mode='lines',
                             line=dict(color=edge_color, width=1.5)
                             )

        data = [triangles, lines]

    else:
        data = [triangles]

    if vlines:
        data.append(vlines)

    # line_marker = dict(color='#0066FF', width=2)

    noaxis = dict(showbackground=False,
                  showline=False,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title='')

    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(
            xaxis=noaxis,
            yaxis=noaxis,
            zaxis=noaxis
            )
        )

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

    if not html_output:
        plotly.offline.iplot(fig)
    else:
        plotly.offline.plot(fig)


def plot_tet_mesh(tetra, vfunc=None, plot_edges=None, plot_levels=False, tfunc=None, cutting=None,
                  edge_color='rgb(50,50,50)', html_output=False, width=800, height=800, flatshading=False,
                  xrange=None, yrange=None, zrange=None, showcaxis=False, caxis=None):

    """
    this is a function to plot tetra meshes

    it is essentially a wrapper around the plotTriaMesh function,
    as the tetra mesh will be converted to its tria boundary mesh,
    and only this will be plotted.

    to view the 'interior' of the tetra mesh, one or more cutting
    criteria can be defined as input arguments to this function:

    e.g. cutting=('x<-10') or cutting=('z>=5') or cutting=('f>4')

    where x,y,z represent dimensions 0,1,2 of the vertex array,
    and f represents the vfunc (which cannot be None if f is used
    to define a cutting criterion)

    only tetras whose vertices fulfill all criteria will be considered
    for plotting

    """
    if type(tetra).__name__ is not "TetMesh":
        raise ValueError('plot_tet_mesh works only on TetMesh class')

    # from plotTriaMesh import plotTriaMesh
    # from tria import is_oriented
    # from tetra import tetra_get_boundary_tria, tetra_fix_orientation

    # evaluate cutting criteria
    if cutting is not None:

        # check inputs
        if type(cutting) is not list:
            cutting = [cutting]

        # check if vfunc is defined when functional thresholds are used, otherwise exit
        if any(['f' in x for x in cutting]) and vfunc is None:
            raise ValueError("Need to specify vfunc if \'f\' is used within the \'cutting\' argument, exiting.")

        # create criteria from cutting info
        criteria = "(" + ")&(".join(
            [x.replace('x', 'tetra.v[:,0]').replace('y', 'tetra.v[:,1]').replace('z', 'tetra.v[:,2]').replace('f',
                                                                                                              'vfunc')
             for x in
             cutting]) + ")"

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
    plot_tria_mesh(tria_bnd, vfunc=vfunc, plot_edges=plot_edges, plot_levels=plot_levels, tfunc=tfunc_tria,
                   edge_color=edge_color, html_output=html_output, width=width, height=height, flatshading=flatshading,
                   xrange=xrange, yrange=yrange, zrange=zrange, showcaxis=showcaxis, caxis=caxis)
