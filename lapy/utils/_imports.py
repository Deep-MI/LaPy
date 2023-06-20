"""Handle optional dependency imports.

Inspired from pandas: https://pandas.pydata.org/
"""

import importlib

# A mapping from import name to package name (on PyPI) when the package name
# is different.
INSTALL_MAPPING = {
    "sksparse": "scikit-sparse",
}


def import_optional_dependency(
    name: str,
    extra: str = "",
    raise_error: bool = True,
):
    """Import an optional dependency.
    
    By default, if a dependency is missing an ImportError with a nice message
    will be raised.

    Args:
        name (str): The module name.
        extra (str, optional): Additional text to include in the ImportError message. (Default value = "")
        raise_error (bool, optional): What to do when a dependency is not found.
    * True : Raise an ImportError.
    * False: Return None. (Default value = True)

    Returns:
        Optional[ModuleType]: The imported module when found.
        None is returned when the package is not found and raise_error is
        False.

    Raises:
        ImportError: dependency not found; see raise_error

    
    """

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_error:
            raise ImportError(
                f"Missing optional dependency '{install_name}'. {extra} "
                f"Use pip or conda to install {install_name}."
            )
        else:
            return None

    return module
