"""EarthOne interaction and utilities"""

import earthdaily.earthone as eo


def get_product_or_fail(product_id: str) -> eo.catalog.Product:
    """A throwing version of eo.catalog.Product.get()

    Parameters
    ----------
    product_id : str
        ID of the product

    Returns
    -------
    eo.catalog.Product
        The requested catalog product
    """

    prod = eo.catalog.Product.get(product_id)
    if prod is None:
        err_msg = (
            f"Product with id '{product_id}' either does not "
            "exist or you do not have access to it"
        )
        raise eo.exceptions.NotFoundError(err_msg)

    return prod
