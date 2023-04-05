from typing import Dict, Union

import descarteslabs as dl
import shapely  # type: ignore
import utm  # type: ignore
from pyproj.crs import CRS


def create_aoi(geometry: Dict, resolution: Union[float, int]):
    """Create an AOI object from a GeoJSON geometry and a resolution (in meters).

    Parameters
    ----------
    geometry : dict
        Geometry to create an AOI for.
    resolution : float, int
        AOI pixel resolution, in meters.

    Returns
    -------
    aoi : dl.geo.AOI
        AOI object.
    """
    # Get the appropriate UTM zone for this polygon
    polygon = shapely.geometry.shape(geometry)
    centroid = polygon.centroid
    _, _, zone_number, zone_letter = utm.from_latlon(centroid.y, centroid.x)
    south = zone_letter < "N"

    # Create a CRS for this UTM zone
    wkt = CRS.from_dict({"proj": "utm", "zone": zone_number, "south": south}).to_wkt()

    aoi = dl.geo.AOI(
        geometry=geometry,
        crs=wkt,
        resolution=resolution,
    )

    return aoi
