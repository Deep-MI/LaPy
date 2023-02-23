import numpy as np


def import_vfunc_deprecated(infile):
    """
    Imports vertex function from txt file. Values can be separated by ; or ,
    and surrounded by {} or () brackets. Also first line can have the
    keyword "Solution:", i.e. the PSOL format from ShapeDNA
    """
    import re

    try:
        f = open(infile, "r")
    except IOError:
        print("[File " + infile + " not found or not readable]")
        return
    txt = f.readlines()
    i = 0
    vals = list()
    while i < len(txt):
        if "Solution:" in txt[i]:
            i = i + 1
            tmp1 = list()
            while i < len(txt):
                if txt[i].isspace():
                    break
                tmp1.append(txt[i].strip())
                i = i + 1
            for tmp2 in re.split("[;,]", re.sub("[{()}]", "", "".join(tmp1))):
                vals.append(float(tmp2))
            # del (tmp1, tmp2)
        i = i + 1
    return vals


def import_vfunc(filename):
    """
    Imports vertex function from txt file. Values can be separated by ; or ,
    and surrounded by {} or () brackets. Also first line can have the
    keyword "Solution:", i.e. the PSOL format from ShapeDNA
    """

    import re

    import numpy as np

    try:
        with open(filename) as f:
            txt = f.readlines()
    except IOError:
        print("[File " + infile + " not found or not readable]")
        return

    txt = [x.strip() for x in txt]

    txt.remove("Solution:")

    txt = [re.sub("[{()}]", "", x) for x in txt]

    if len(txt) == 1:
        txt = [re.split("[,;]", x) for x in txt][0]

    txt = [np.float(x) for x in txt]

    # txt = np.array(txt)

    return txt


def import_ev(infile):
    """
    Load EV file
    """
    # open file
    try:
        f = open(infile, "r")
    except IOError:
        print("[File " + infile + " not found or not readable]")
        return
    # read file (and get rid of all \n)
    ll = f.read().splitlines()
    # define data structure
    d = dict()
    # go through each line and parse it
    i = 0
    while i < len(ll):
        if ll[i].lstrip().startswith("Creator:"):
            d.update({"Creator": ll[i].split(":", 1)[1].strip()})
            i = i + 1
        elif ll[i].lstrip().startswith("File:"):
            d.update({"File": ll[i].split(":", 1)[1].strip()})
            i = i + 1
        elif ll[i].lstrip().startswith("User:"):
            d.update({"User": ll[i].split(":", 1)[1].strip()})
            i = i + 1
        elif ll[i].lstrip().startswith("Refine:"):
            d.update({"Refine": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Degree:"):
            d.update({"Degree": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Dimension:"):
            d.update({"Dimension": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Elements:"):
            d.update({"Elements": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("DoF:"):
            d.update({"DoF": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("NumEW:"):
            d.update({"NumEW": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Area:"):
            d.update({"Area": float(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Volume:"):
            d.update({"Volume": float(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("BLength:"):
            d.update({"BLength": float(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("EulerChar:"):
            d.update({"EulerChar": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Time(pre)"):
            d.update({"TimePre": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Time(calcAB)"):
            d.update({"TimeCalcAB": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Time(calcEW)"):
            d.update({"TimeCalcEW": int(ll[i].split(":", 1)[1].strip())})
            i = i + 1
        elif ll[i].lstrip().startswith("Eigenvalues"):
            i = i + 1
            while ll[i].find("{") < 0:  # possibly introduce termination criterion
                i = i + 1
            if ll[i].find("}") >= 0:  # '{' and '}' on the same line
                evals = ll[i].strip().replace("{", "").replace("}", "")
            else:
                evals = str()
                while ll[i].find("}") < 0:
                    evals = evals + ll[i].strip().replace("{", "").replace("}", "")
                    i = i + 1
                evals = evals + ll[i].strip().replace("{", "").replace("}", "")
            evals = np.array(evals.split(";")).astype(np.float)
            d.update({"Eigenvalues": evals})
            i = i + 1
        elif ll[i].lstrip().startswith("Eigenvectors"):
            i = i + 1
            while not (ll[i].strip().startswith("sizes")):
                i = i + 1
            d.update(
                {"EigenvectorsSize": np.array(ll[i].strip().split()[1:]).astype(np.int)}
            )
            i = i + 1
            while ll[i].find("{") < 0:  # possibly introduce termination criterion
                i = i + 1
            if ll[i].find("}") >= 0:  # '{' and '}' on the same line
                evecs = ll[i].strip().replace("{", "").replace("}", "")
            else:
                evecs = str()
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
            ).astype(np.float)
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


def export_ev(outfile, d):
    """
    Save EV data structures as txt file (format from ShapeDNA)
    usage: exportEV(data,outfile)
    """
    # open file
    try:
        f = open(outfile, "w")
    except IOError:
        print("[File " + outfile + " not writable]")
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


def export_vfunc(outfile, vfunc):
    """
    Exports vertex function in PSOL txt file:
    First line "Solution:", "," separated values inside ()
    """
    try:
        f = open(outfile, "w")
    except IOError:
        print("[File " + outfile + " not writable]")
        return
    f.write("Solution:\n")
    f.write("(" + ",".join(vfunc.astype(str)) + ")")
    f.close()
