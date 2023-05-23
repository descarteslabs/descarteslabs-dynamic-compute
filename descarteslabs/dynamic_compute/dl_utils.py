"""Descartes Labs interaction and utilities"""

import descarteslabs as dl


def get_product_or_fail(product_id: str) -> dl.catalog.Product:
    """A throwing version of dl.catalog.Product.get()

    Parameters
    ----------
    product_id : str
        ID of the product

    Returns
    -------
    dl.catalog.Product
        The requested catalog product
    """

    prod = dl.catalog.Product.get(product_id)
    if prod is None:
        err_msg = (
            f"Product with id '{product_id}' either does not "
            "exist or you do not have access to it"
        )
        raise dl.exceptions.NotFoundError(err_msg)

    return prod
