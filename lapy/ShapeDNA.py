"""

shapeDNA - a script to compute ShapeDNA of FreeSurfer structures

SUMMARY
=======

Computes surface (and volume) models of FreeSurfer's subcortical and cortical
structures and computes a spectral shape descriptor (ShapeDNA [1]).

Required arguments are:

'sid'     : subject ID
'sdir'    : subjects directory

Input can be (exactly) one of the following:

'asegid'  : an aseg label id to construct a surface of that ROI
'hsfid'   : a hippocampal subfields id to construct a surface of that ROI (also pass 'hemi')
'surf'    : a surface, e.g. lh.white

Information about optional arguments can be optained with:

'help'    : display help message and usage info

Note that FreeSurfer needs to be sourced for this program to run.


ARGUMENTS
=========

REQUIRED ARGUMENTS

'sid' <name>      Subject ID

'sdir' <name>     Subjects directory

One of the following:

'asegid' <int>    Segmentation ID of structure in aseg.mgz (e.g. 17 is
                  Left-Hippocampus), for ID's check <sid>/stats/aseg.stats
                  or $FREESURFER_HOME/FreeSurferColorLUT.txt.

'hsfid' <int>     Segmentation ID of structure in
                  [lr]h.<segmentation>Labels-<label>.<version><suffix>
                  For ID's (e.g. 206 is CA1) check compressionLookupTable.txt
                  in $FREESURFER_HOME/average/HippoSF/atlas. Also specify
                  'hemi' if using this argument.

'surf' <name>     lh.pial, rh.pial, lh.white, rh.white etc. to select a
                  surface from the <sid>/surfs directory.

OPTIONAL ARGUMENT

'outdir' <name>   Output directory (default: <sdir>/<sid>/shapedna/)

REQUIRED ARGUMENT if using 'hsfid'

'hemi' <name>     'lh' or 'rh' hemisphere

OPTIONAL ARGUMENTS if using 'hsfid'

'hsflabel' <name> Label of hippocampal subfield segmentation, e.g. T1 or
                  T1-T1T2_HC_segmentation (default: T1)

'hsfver' <name>   Version of hippocampal subfield segmentation, e.g v10 or
                  v20 (default: v10)

'hsfseg' <name>   Name of segmentation, e.g. hippoSf or hippoAmyg (default:
                  hippoSf)

'hsfsfx' <name>   Suffix of segmenation, e.g. FSvoxelSpace or none (specify
                  as '') (default: FSvoxelSpace)

ShapeDNA parameters:

'num' <int>       Number of eigenvalues/vectors to compute (default: 50)

'bcond' <int>     Boundary condition (0=Dirichlet, 1=Neumann default)

'evec'            Additionally compute eigenvectors


REFERENCES
==========

Always cite [1] as it describes the method. If you use topological features of
eigenfunctions, also cite [2]. [3] compares different discretizations of
Laplace-Beltrami operators and shows that the used FEM appraoch performs best.
If you do statistical shape analysis you may also want to cite [4] as it
discusses medical applications.

[1] M. Reuter, F.-E. Wolter and N. Peinecke.
Laplace-Beltrami spectra as "Shape-DNA" of surfaces and solids.
Computer-Aided Design 38 (4), pp.342-366, 2006.
http://dx.doi.org/10.1016/j.cad.2005.10.011

[2] M. Reuter. Hierarchical Shape Segmentation and Registration
via Topological Features of Laplace-Beltrami Eigenfunctions.
International Journal of Computer Vision, 2009.
http://dx.doi.org/10.1007/s11263-009-0278-1

[3] M. Reuter, S. Biasotti, D. Giorgi, G. Patane, M. Spagnuolo.
Discrete Laplace-Beltrami operators for shape analysis and
segmentation. Computers & Graphics 33 (3), pp.381-390, 2009.
http://dx.doi.org/10.1016/j.cag.2009.03.005

[4] M. Reuter, F.-E. Wolter, M. Shenton, M. Niethammer.
Laplace-Beltrami Eigenvalues and Topological Features of
Eigenfunctions for Statistical Shape Analysis.
Computer-Aided Design 41 (10), pp.739-755, 2009.
http://dx.doi.org/10.1016/j.cad.2009.02.007

"""
# ==============================================================================
# FUNCTIONS

# ------------------------------------------------------------------------------
# auxiliary functions

# function to run commands
def run_cmd(cmd, err_msg):
    """
    execute the command
    """

    # imports
    import os
    import sys
    import shlex
    import subprocess

    # aux function
    def which(program):
        """
        check if executable
        """

        # aux function
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)

        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
            if is_exe(os.path.join('.', program)):
                return os.path.join('.', program)

        return None

    clist = cmd.split()
    progname = which(clist[0])
    if progname is None:
        print('ERROR: ' + clist[0] + ' not found in path!', flush=True)
        sys.exit(1)
    clist[0] = progname
    cmd = ' '.join(clist)
    print('#@# Command: ' + cmd + '\n', flush=True)

    args = shlex.split(cmd)
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        print('ERROR: ' + err_msg, flush=True)
        raise

# ------------------------------------------------------------------------------
# check options

# check_options
def check_options(options):
    """
    checks if input options are valid, and sets some defaults
    """

    # imports
    import os
    import sys
    import tempfile
    import uuid
    import errno

    # check if Freesurfer has been sourced
    fshome = os.getenv('FREESURFER_HOME')
    if fshome is None:
        print('\nERROR: Environment variable FREESURFER_HOME not set. \n', flush=True)
        print('       Make sure to source your FreeSurfer installation.\n', flush=True)
        sys.exit(1)

    # subjects dir must be given
    if options["sdir"] is None:
        print('\nERROR: specify subjects directory via --sdir\n', flush=True)
        sys.exit(1)

    # subject id must be given and exist in subjects dir
    if options["sid"] is None:
        print('\nERROR: Specify --sid\n', flush=True)
        sys.exit(1)

    subjdir = os.path.join(options["sdir"], options["sid"])
    if not os.path.exists(subjdir):
        print('ERROR: cannot find sid in subjects directory\n', flush=True)
        sys.exit(1)

    # input needs to be either a surf or aseg or hipposf label(s)
    if options["asegid"] is None and options["surf"] is None and options["hsfid"] is None:
        print('\nERROR: Specify either --asegid or --hsfid or --surf\n', flush=True)
        sys.exit(1)

    # and it cannot be more than one of them
    if sum((options["asegid"] is not None, options["surf"] is not None, options["hsfid"] is not None)) > 1:
        print('\nERROR: Specify either --asegid or --hsfid or --surf (not more than one of them)\n', flush=True)
        sys.exit(1)

    # if --hsfid is used, then also --hemi needs to be specified
    if options["hsfid"] is not None and options["hemi"] is None:
        print('\nERROR: Specify --hemi\n', flush=True)
        sys.exit(1)

    # hemi needs to be either lh or rh
    if options["hemi"] is not None and options["hemi"] != "lh" and options["hemi"] != "rh":
        print('\nERROR: Specify either --hemi=lh or --hemi=rh\n', flush=True)
        sys.exit(1)

    # check suffix to start with a point
    if options["hsfsfx"] is not None and options["hsfsfx"][0] is not ".":
        options["hsfsfx"] = "." + options["hsfsfx"]
        print(options["hsfsfx"])

    # check if required files are present for --hsfid
    if options["hsfid"] is not None and options["hemi"] == "lh" and not os.path.isfile(os.path.join(options["sdir"], options["sid"], 'mri', 'lh.' + options["hsfseg"] + 'Labels-' + options["hsflabel"] + '.' + options["hsfver"] + options["hsfsfx"] + '.mgz')):
        print('\nERROR: could not find ' + os.path.join(options["sdir"], options["sid"], 'mri', 'lh.' + options["hsfseg"] + 'Labels-' + options["hsflabel"] + '.' + options["hsfver"] + options["hsfsfx"] + '.mgz') + '\n', flush=True)
        sys.exit(1)

    if options["hsfid"] is not None and options["hemi"] == "rh" and not os.path.isfile(os.path.join(options["sdir"], options["sid"], 'mri', 'rh.' + options["hsfseg"] + 'Labels-' + options["hsflabel"] + '.' + options["hsfver"] + options["hsfsfx"] + '.mgz')):
        print('\nERROR: could not find ' + os.path.join(options["sdir"], options["sid"], 'mri', 'rh.' + options["hsfseg"] + 'Labels-' + options["hsflabel"] + '.' + options["hsfver"] + options["hsfsfx"] + '.mgz') + '\n', flush=True)
        sys.exit(1)

    # set default output dir
    if options["outdir"] is None:
        options["outdir"] = os.path.join(subjdir, 'shapedna')
    try:
        os.mkdir(options["outdir"])
    except OSError as e:
        if e.errno != errno.EEXIST:  # directory exists (but that's OK)
            raise e
        pass

    # check if we have write access to output dir
    try:
        testfile = tempfile.TemporaryFile(dir=options["outdir"])
        testfile.close()
    except OSError as e:
        if e.errno != errno.EACCES:  # 13
            e.filename = options["outdir"]
            raise
        print('\nERROR: ' + options["outdir"] + ' not writeable (check access)!\n', flush=True)
        sys.exit(1)

    # initialize outsurf
    if options["asegid"] is not None:
        astring = '_'.join(options["asegid"])
        surfname = 'aseg.' + astring + '.vtk'
        options["outsurf"] = os.path.join(options["outdir"], surfname)
    elif options["hsfid"] is not None:
        astring = str(options["hemi"]) + '-' + '_'.join(options["hsfid"])
        surfname = 'hsf.' + astring + '.vtk'
        options["outsurf"] = os.path.join(options["outdir"], surfname)
    else:
        # for other surfaces, a refined/smoothed version could be written
        surfname = os.path.basename(options["surf"]) + '.vtk'
        options["outsurf"] = os.path.join(options["outdir"], surfname)

    return options

# ------------------------------------------------------------------------------
# image and surface processing functions

# creates a surface from the aseg and label info and writes it to the outdir
def get_aseg_surf(options):
    """
    a function to create a surface from the aseg and label files
    """

    # imports
    import os
    import uuid

    #
    astring = ' '.join(options["asegid"])
    subjdir = os.path.join(options["sdir"], options["sid"])
    aseg = os.path.join(subjdir, 'mri', 'aseg.mgz')
    norm = os.path.join(subjdir, 'mri', 'norm.mgz')
    tmpname = 'aseg.' + str(uuid.uuid4())
    segf = os.path.join(options["outdir"], tmpname + '.mgz')
    segsurf = os.path.join(options["outdir"], tmpname + '.surf')
    # binarize on selected labels (creates temp segf)
    # always binarize first, otherwise pretess may scale aseg if labels are larger than 255 (e.g. aseg+aparc, bug in mri_pretess?)
    cmd = 'mri_binarize --i ' + aseg + ' --match ' + astring + ' --o ' + segf
    run_cmd(cmd, 'mri_binarize failed.')
    ptinput = segf
    ptlabel = '1'
    # if norm exist, fix label (pretess)
    if os.path.isfile(norm):
        cmd = 'mri_pretess ' + ptinput + ' ' + ptlabel + ' ' + norm + ' ' + segf
        run_cmd(cmd, 'mri_pretess failed.')
    else:
        if not os.path.isfile(segf):
            # cp segf if not exist yet (it exists already if we combined labels above)
            cmd = 'cp ' + ptinput + ' ' + segf
            run_cmd(cmd, 'cp segmentation file failed.')
    # runs marching cube to extract surface
    cmd = 'mri_mc ' + segf + ' ' + ptlabel + ' ' + segsurf
    run_cmd(cmd, 'mri_mc failed?')
    # convert to vtk
    cmd = 'mris_convert ' + segsurf + ' ' + options["outsurf"]
    run_cmd(cmd, 'mris_convert failed.')
    # return surf name
    return options["outsurf"]

# creates a surface from the hsf and label info and writes it to the outdir
def get_hsf_surf(options):
    """
    a function to create surfaces from hsf masks and labels
    """

    # imports
    import os
    import sys
    import uuid

    #
    astring = ' '.join(options["hsfid"])
    subjdir = os.path.join(options["sdir"], options["sid"])
    hsf = os.path.join(options["sdir"], options["sid"], 'mri', options["hemi"] + '.' + options["hsfseg"] + 'Labels-' + options["hsflabel"] + '.' + options["hsfver"] + options["hsfsfx"] + '.mgz')
    print('Creating surfaces from ' + hsf, flush=True)
    if not os.path.isfile(hsf):
        print('\nERROR: could not open ' + hsf + "\n", flush=True)
        sys.exit(1)
    norm = os.path.join(subjdir, 'mri', 'norm.mgz')
    tmpname = 'hsf.' + str(uuid.uuid4())
    segf = os.path.join(options["outdir"], tmpname + '.mgz')
    segsurf = os.path.join(options["outdir"], tmpname + '.surf')
    # binarize on selected labels (creates temp segf)
    # always binarize first, otherwise pretess may scale aseg if labels are larger than 255 (e.g. aseg+aparc, bug in mri_pretess?)
    cmd = 'mri_binarize --i ' + hsf + ' --match ' + astring + ' --o ' + segf
    run_cmd(cmd, 'mri_binarize failed.')
    ptinput = segf
    ptlabel = '1'
    # if norm exist, fix label (pretess)
    if os.path.isfile(norm):
        cmd = 'mri_pretess ' + ptinput + ' ' + ptlabel + ' ' + norm + ' ' + segf
        run_cmd(cmd, 'mri_pretess failed.')
    else:
        if not os.path.isfile(segf):
            # cp segf if not exist yet (it exists already if we combined labels above)
            cmd = 'cp ' + ptinput + ' ' + segf
            run_cmd(cmd, 'cp segmentation file failed.')
    # runs marching cube to extract surface
    cmd = 'mri_mc ' + segf + ' ' + ptlabel + ' ' + segsurf
    run_cmd(cmd, 'mri_mc failed?')
    # convert to vtk
    cmd = 'mris_convert ' + segsurf + ' ' + options["outsurf"]
    run_cmd(cmd, 'mris_convert failed.')
    # return surf name
    return options["outsurf"]

# gets global path to surface input (if it is a FS surf)
def get_surf_surf(options):
    """
    a function to return a surface name
    """

    # imports
    import os

    # check if surface has VTK format
    if (os.path.splitext(options["surf"])[1]).upper() != '.VTK':
        print("Converting surface "+options["surf"]+" to vtk format ...")
        cmd = 'mris_convert ' + options["surf"] + ' ' + options["outsurf"]
        run_cmd(cmd, 'mris_convert failed.')

    # return surf name
    return options["outsurf"]

# ------------------------------------------------------------------------------
# shapeDNA functions

# run shapeDNA-tria
def compute_shapeDNA_tria(surf, options):
    """
    a function to run the shapeDNA / laplaceTria script
    """

    # imports
    from lapy import TriaIO, FuncIO, Solver

    # get surface
    tria = TriaIO.import_vtk(surf)

    # get fem, evals, evecs
    fem = Solver(tria)
    evals, evecs = fem.eigs(k=3)

    # write ev

    evDict = dict()
    evDict['Refine'] = 0
    evDict['Degree'] = 1
    evDict['Dimension'] = 2
    evDict['Elements'] = len(tria.t)
    evDict['DoF'] = len(tria.v)
    evDict['NumEW'] = options["num"]
    evDict['Eigenvalues'] = evals
    evDict['Eigenvectors'] = evecs

    outev = options["outsurf"] + '.ev'

    FuncIO.export_ev(outev, evDict)

    # write surf

    TriaIO.export_vtk(tria, options["outsurf"])

    # return

    return tria, evals, evecs

# ------------------------------------------------------------------------------
# main function

# compute_ShapeDNA
def compute_ShapeDNA(sid=None, sdir=None, outdir=None, asegid=None, hsfid=None, surf=None, hemi=None, hsflabel=None, hsfver=None, hsfseg=None, hsfsfx=None, num=50, bcond=1, evec=False):

    # imports
    import os
    import sys
    import warnings

    # get options
    options = dict(sid=sid, sdir=sdir, outdir=outdir, asegid=asegid, hsfid=hsfid, surf=surf, hemi=hemi, hsflabel=hsflabel, hsfver=hsfver, hsfseg=hsfseg, hsfsfx=hsfsfx, num=num, bcond=bcond, evec=evec)

    # check command line options
    options = check_options(options)

    # get surface
    if options["asegid"] is not None:
        surf = get_aseg_surf(options)
        outsurf = surf
    elif options["hsfid"] is not None:
        surf = get_hsf_surf(options)
        outsurf = surf
    elif options["surf"] is not None:
        surf = get_surf_surf(options)
        outsurf = options["outsurf"]
    else:
        surf = None

    if surf is None:
        print('\nERROR: no surface was created/selected?\n', flush=True)
        sys.exit(1)

    # run shapeDNA tria
    tria, evals, evecs = compute_shapeDNA_tria(surf, options)
