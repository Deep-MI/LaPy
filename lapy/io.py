"""Functions to read and write spectra and vertex functions."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def read_vfunc(filename):
    """Import vertex functions from txt file.

    Values can be separated by ``;`` or ``,`` and surrounded by ``{}`` or ``()``
    brackets. Also first line can have the keyword "Solution:", i.e. the PSOL format
    from ShapeDNA.

    Parameters
    ----------
    filename : str
        Filename of input.

    Returns
    -------
    vals : array
        List of vfunc parameters.
    """
    import re

    try:
        with open(filename) as f:
            txt = [x.strip() for x in f]
    except OSError:
        logger.error("File %s not found or not readable", filename)
        raise

    if "Solution:" not in txt:
        raise ValueError(f"Expected 'Solution:' marker in {filename}")
    txt.remove("Solution:")
    txt = [re.sub("[{()}]", "", x) for x in txt if x]
    if not txt:
        raise ValueError(f"No vertex function data found in {filename}")

    if len(txt) == 1:
        txt = [re.split("[,;]", x) for x in txt][0]
    return [float(x) for x in txt]


def read_ev(filename):
    """Load EV file.

    Parameters
    ----------
    filename : str
        Filename of input.

    Returns
    -------
    d: dict
        Dictionary of eigenvalues, eigenvectors (optional), and associated
        information.
    """
    # open file
    try:
        with open(filename) as f:
            # read file (and get rid of all \n)
            ll = f.read().splitlines()
    except OSError:
        logger.error("File %s not readable", filename)
        raise

    # define data structure
    d = {}
    # go through each line and parse it
    i = 0

    def _parse_field(label, cast=int):
        nonlocal i
        d[label] = cast(ll[i].split(":", 1)[1].strip())
        i += 1

    while i < len(ll):
        line = ll[i].lstrip()
        if ":" in line:
            key, _ = line.split(":", 1)
            if key in {
                "Creator",
                "File",
                "User",
                "Refine",
                "Degree",
                "Dimension",
                "Elements",
                "DoF",
                "NumEW",
                "Area",
                "Volume",
                "BLength",
                "EulerChar",
                "Time(pre)",
                "Time(calcAB)",
                "Time(calcEW)",
            }:
                _parse_field(key if key != "Time(pre)" else "TimePre",
                             float if key in {"Area", "Volume", "BLength"} else int)
                continue

        if line.startswith("Eigenvalues"):
            i = i + 1
            while ll[i].find("{") < 0:  # possibly introduce termination criterion
                i = i + 1
            if ll[i].find("}") >= 0:  # '{' and '}' on the same line
                evals = ll[i].strip().replace("{", "").replace("}", "")
            else:
                evals = ""
                while ll[i].find("}") < 0:
                    evals = evals + ll[i].strip().replace("{", "").replace("}", "")
                    i = i + 1
                evals = evals + ll[i].strip().replace("{", "").replace("}", "")
            evals = np.array(evals.split(";")).astype(float)
            d.update({"Eigenvalues": evals})
            i = i + 1
        elif line.startswith("Eigenvectors"):
            i = i + 1
            while not (ll[i].strip().startswith("sizes")):
                i = i + 1
            d.update(
                {"EigenvectorsSize": np.array(ll[i].strip().split()[1:]).astype(int)}
            )
            i = i + 1
            while ll[i].find("{") < 0:  # possibly introduce termination criterion
                i = i + 1
            if ll[i].find("}") >= 0:  # '{' and '}' on the same line
                evecs = ll[i].strip().replace("{", "").replace("}", "")
            else:
                evecs = ""
                while ll[i].find("}") < 0:
                    evecs = evecs + ll[i].strip().replace("{", "").replace(
                        "}", ""
                    ).replace("(", "").replace(")", "")
                    i = i + 1
                evecs = evecs + ll[i].strip().replace("{", "").replace("}", "").replace(
                    "(", ""
                ).replace(")", "")
            evecs = np.array(
                evecs.replace(";", " ").replace(",", " ").strip().split()
            ).astype(float)
            if len(evecs) == (d["EigenvectorsSize"][0] * d["EigenvectorsSize"][1]):
                evecs = np.transpose(np.reshape(evecs, d["EigenvectorsSize"][1::-1]))
                d.update({"Eigenvectors": evecs})
            else:
                print(
                    "[Length of eigenvectors is not "
                    + str(d["EigenvectorsSize"][0])
                    + " times "
                    + str(d["EigenvectorsSize"][1])
                    + "."
                )
            i = i + 1
        else:
            i = i + 1
    # close file
    f.close()
    # return dict
    return d


def write_ev(filename, d):
    """Save EV data structures as txt file (format from ShapeDNA).

    Parameters
    ----------
    filename : str
        Filename to save to.
    d : dict
        Dictionary of eigenvalues, eigenvectors (optional), and associated
        information.
    """
    # open file
    try:
        f = open(filename, "w")
    except OSError:
        print("[File " + filename + " not writable]")
        return
    # check data structure
    if "Eigenvalues" not in d:
        print("ERROR: no Eigenvalues specified")
        exit(1)
    # ...
    # Write
    if "Creator" in d:
        f.write(" Creator: " + d["Creator"] + "\n")
    if "File" in d:
        f.write(" File: " + d["File"] + "\n")
    if "User" in d:
        f.write(" User: " + d["User"] + "\n")
    if "Refine" in d:
        f.write(" Refine: " + str(d["Refine"]) + "\n")
    if "Degree" in d:
        f.write(" Degree: " + str(d["Degree"]) + "\n")
    if "Dimension" in d:
        f.write(" Dimension: " + str(d["Dimension"]) + "\n")
    if "Elements" in d:
        f.write(" Elements: " + str(d["Elements"]) + "\n")
    if "DoF" in d:
        f.write(" DoF: " + str(d["DoF"]) + "\n")
    if "NumEW" in d:
        f.write(" NumEW: " + str(d["NumEW"]) + "\n")
    f.write("\n")
    if "Area" in d:
        f.write(" Area: " + str(d["Area"]) + "\n")
    if "Volume" in d:
        f.write(" Volume: " + str(d["Volume"]) + "\n")
    if "BLength" in d:
        f.write(" BLength: " + str(d["BLength"]) + "\n")
    if "EulerChar" in d:
        f.write(" EulerChar: " + str(d["EulerChar"]) + "\n")
    f.write("\n")
    if "TimePre" in d:
        f.write(" Time(Pre) : " + str(d["TimePre"]) + "\n")
    if "TimeCalcAB" in d:
        f.write(" Time(calcAB) : " + str(d["TimeCalcAB"]) + "\n")
    if "TimeCalcEW" in d:
        f.write(" Time(calcEW) : " + str(d["TimeCalcEW"]) + "\n")
    if "TimePre" in d and "TimeCalcAB" in d and "TimeCalcEW" in d:
        f.write(
            " Time(total ) : "
            + str(d["TimePre"] + d["TimeCalcAB"] + d["TimeCalcEW"])
            + "\n"
        )
    f.write("\n")
    f.write("Eigenvalues:\n")
    f.write(
        "{ " + " ; ".join(map(str, d["Eigenvalues"])) + " }\n"
    )  # consider precision
    f.write("\n")
    if "Eigenvectors" in d:
        f.write("Eigenvectors:\n")
        # f.write('sizes: '+' '.join(map(str,d['EigenvectorsSize']))+'\n')
        # better compute real sizes from eigenvector array?
        f.write("sizes: " + " ".join(map(str, d["Eigenvectors"].shape)) + "\n")
        f.write("\n")
        f.write("{ ")
        for i in range(np.shape(d["Eigenvectors"])[1] - 1):
            f.write("(")
            f.write(",".join(map(str, d["Eigenvectors"][:, i])))
            f.write(") ;\n")
        f.write("(")
        f.write(
            ",".join(
                map(
                    str,
                    d["Eigenvectors"][:, np.shape(d["Eigenvectors"])[1] - 1],
                )
            )
        )
        f.write(") }\n")
    # close file
    f.close()


def write_vfunc(filename, vfunc):
    """Save vertex in PSOL txt file.

    First line "Solution:", "," separated values inside ()

    Parameters
    ----------
    filename : str
        Filename to save to.
    vfunc : array_like
        List of vfunc parameters.
    """
    try:
        f = open(filename, "w")
    except OSError:
        print("[File " + filename + " not writable]")
        return
    f.write("Solution:\n")
    f.write("(" + ",".join(vfunc.astype(str)) + ")")
    f.close()
