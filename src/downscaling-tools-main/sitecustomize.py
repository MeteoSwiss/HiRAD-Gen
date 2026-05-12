"""Runtime compatibility shims for the local downscaling-tools environment.

This repo is placed on ``PYTHONPATH`` by the evaluation job templates, so
``sitecustomize`` is imported automatically on interpreter startup. Keep any
patches here narrow and safe.
"""


def _patch_earthkit_array_compat() -> None:
    """Restore legacy helpers expected by earthkit.data.

    The current local environment ships a newer ``earthkit.utils.array`` API
    that exposes ``array_namespace`` and ``convert`` but no longer exports the
    legacy ``array_to_numpy``, ``convert_array``, or ``get_backend`` helpers.
    ``earthkit.data`` still imports those symbols, which breaks the TC plotting
    path. Recreate the old entry points in terms of the new API.
    """

    import numpy as np

    import earthkit.utils.array as eka
    from earthkit.utils.array.array_namespace import array_namespace
    from earthkit.utils.array.convert import convert

    if not hasattr(eka, "array_to_numpy"):
        def array_to_numpy(array, *args, **kwargs):
            del args, kwargs
            return np.asarray(convert(array, array_namespace="numpy"))

        eka.array_to_numpy = array_to_numpy

    if not hasattr(eka, "get_backend"):
        def get_backend(obj=None, *args, **kwargs):
            del args, kwargs
            if obj is None:
                return array_namespace("numpy")
            return array_namespace(obj)

        eka.get_backend = get_backend

    if not hasattr(eka, "convert_array"):
        def convert_array(array, target_backend=None, target_array_sample=None, **kwargs):
            if target_backend is None and target_array_sample is not None:
                target_backend = array_namespace(target_array_sample)
            if target_backend is None:
                return array
            return convert(array, array_namespace=target_backend, **kwargs)

        eka.convert_array = convert_array


try:
    _patch_earthkit_array_compat()
except Exception:
    # Do not block unrelated commands if earthkit is unavailable.
    pass
