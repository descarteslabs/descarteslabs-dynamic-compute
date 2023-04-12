def validate_scales(scales):
    """
    Validate and normalize a list of scales for an XYZ layer.

    A _scaling_ is a 2-tuple (or 2-list) like ``[min, max]``,
    meaning the range of values in your data you want to stretch
    to the 0..255 output range.

    If ``min`` and ``max`` are ``None``, the min/max values in the
    data will be used automatically. Since each tile is computed
    separately and may have different min/max values, this can
    create a "patchwork" effect when viewed on a map.

    Scales can be given as:

    * Three scalings, for 3-band images, like ``[[0, 1], [0, 0.5], [None, None]]``
    * Two scalings, for 2-band images, like ``[[0, 1], [None, None]]``
    * 1-list/tuple of 1 scaling, for 1-band images, like ``[[0, 1]]``
    * 1 scaling (for convenience), which is equivalent to the above: ``[0, 1]``
    * None, or an empty list or tuple for no scalings

    Parameters
    ----------
    scales: list, tuple, or None
        The scales to validate, in the format shown above

    Returns
    -------
    scales: list
        0- to 3-length list of scalings, where each item is a float.
        (``[0, 1]`` would become ``[[0.0, 1.0]]``, for example.)
        If no scalings are given, an empty list is returned.

    Raises
    ------
    TypeError, ValueError
        If the scales do not match the correct format
    """
    if scales is not None:
        if not isinstance(scales, (list, tuple)):
            raise TypeError(
                "Expected a list or tuple of scales, but got {}".format(scales)
            )

        if (
            len(scales) == 2
            and not isinstance(scales[0], (list, tuple))
            and not isinstance(scales[1], (list, tuple))
        ):
            # allow a single 2-tuple for convenience with colormaps/1-band images
            scales = (scales,)

        if len(scales) > 3:
            raise (
                ValueError(
                    "Too many scales passed: expected up to 3 scales, but got {}".format(
                        len(scales)
                    )
                )
            )

        for i, scaling in enumerate(scales):
            if not isinstance(scaling, (list, tuple)):
                raise TypeError(
                    "Scaling {}: expected a 2-item list or tuple for the scaling, "
                    "but got {}".format(i, scaling)
                )
            if len(scaling) != 2:
                raise ValueError(
                    "Scaling {}: expected a 2-item list or tuple for the scaling, "
                    "but length was {}".format(i, len(scaling))
                )
            if not all(isinstance(x, (int, float, type(None))) for x in scaling):
                raise TypeError(
                    "Scaling {}: items in scaling must be numbers or None; "
                    "got {}".format(i, scaling)
                )
            # At this point we know they are all int, float, or None
            # So we check to see if we have an int/float and a None
            if any(isinstance(x, (int, float)) for x in scaling) and any(
                x is None for x in scaling
            ):
                raise ValueError(
                    "Invalid scales passed: one number and one None in scales[{}] {}".format(
                        i, scaling
                    )
                )

        return [
            [float(x) if isinstance(x, int) else x for x in scaling]
            for scaling in scales
        ]
        # be less strict about floats than traitlets is
    else:
        return []
