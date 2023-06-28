# Dynamic-Compute 🗺️

![PyPI](https://img.shields.io/pypi/v/descarteslabs-dynamic-compute)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/descarteslabs-dynamic-compute)
![PyPI - License](https://img.shields.io/pypi/l/descarteslabs-dynamic-compute)

> "It occurs to me that our survival may depend upon our talking to one another." — "Sol Weintraub", [Hyperion](<https://en.wikipedia.org/wiki/Hyperion_(Simmons_novel)>)

_Dynamic-Compute_ is a **map computation engine**. It enables users to **dynamically** generate maps from a **composable** set of Python operations. Together, these properties enable data scientists in the building of complex GIS applications.

Formal documentation for this library is available under the [Descartes Labs API Documentation](https://docs.descarteslabs.com/api.html).

Example notebooks to get started can be found under [Descartes Labs Guides](https://docs.descarteslabs.com/guide.html). Below is a very simple example to get you started using the map:

First, we import `descarteslabs.dynamic_compute` and instantiate the map, then set the zoom level and lat, long of the center:

```python
import descarteslabs.dynamic_compute as dc

m = dc.map
m.zoom = 14
m.center = (43.4783, -110.7506)
m
```

Next, we can create a layer from a Descartes Labs Catalog product by executing the following Python code:

```python
spot_rgb = (
    dc.Mosaic.from_product_bands(
        "airbus:oneatlas:spot:v2",
        "red green blue",
        start_datetime="20210101",
        end_datetime="2022101",
    )
)
```

We can then visualize this on the map using by calling `.visualize` on our layer:

```python
_ = spot_rgb.visualize("SPOT", m, scales=[[0, 256], [0, 256], [0, 256]])
```

Only files included in `__all__` will be supported.
