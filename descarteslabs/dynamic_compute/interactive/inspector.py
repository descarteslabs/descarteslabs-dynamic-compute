import threading
import weakref

import cachetools
import ipyleaflet
import ipywidgets
import numpy as np
import traitlets

from ..operations import value_at
from .layer import DynamicComputeLayer


class JobTimeoutError(Exception):
    "Raised when a computation took longer to complete than a specified timeout."
    pass


class CircleMarkerWithLatLon(ipyleaflet.CircleMarker):
    """
    CircleMarker that offers an XYZ-tile GeoContext containing its current location.

    Just used as a place to cache the GeoContext so each InspectorRowGenerator
    doesn't have to recompute it.
    """

    map = traitlets.Instance(ipyleaflet.Map, allow_none=True)
    inspect_latlon = traitlets.Tuple(allow_none=True, read_only=True)

    @traitlets.observe("location", "map", type="change")
    def _update_geoctx(self, change):
        if self.map is None:
            with self.hold_trait_notifications():
                self.set_trait("inspect_latlon", ())
            return

        with self.hold_trait_notifications():
            self.set_trait("inspect_latlon", self.location)


class PixelInspector(ipywidgets.GridBox):
    """
    Display pixel values when clicking on the map.

    Whenever you click on the map, it fetches the pixel values at that
    location for all active Workflows layers and displays them in a table
    overlaid on the map. It also shows a marker on the map indicating
    the last position clicked. As layers change, or are added or removed,
    the table keeps fetching pixel values for the new layers at the last-clicked
    point (the marker's current position).


    To unlink from the map, call `~.unlink`.

    Example
    -------
    >>> import descarteslabs.dynamic_compute as dc
    >>> my_map = dc.map
    >>> naip_rgb = dc.Mosaic.from_product_bands("usda:naip:rgbn:v1", "red green blue") # doctest: +SKIP
                        .visualize( "NAIP", m, scales=[[0, 1], [0, 1], [0, 1]]) # doctest: +SKIP
    >>> inspector = dc.interactive.PixelInspector(my_map)
    >>> my_map  # doctest: +SKIP
    >>> # ^ display the map
    >>> # click on the map; a table will pop up showing pixel values for each band in the NAIP layer
    >>> inspector.unlink()
    >>> # table and marker disappear; click again and nothing happens
    """

    # NOTE(gabe): we use the marker's opacity as a crude global on-off switch;
    # all event listeners check that opacity == 1 before doing actual work.
    marker = traitlets.Instance(
        CircleMarkerWithLatLon,
        kw=dict(opacity=0, radius=5, weight=1, name="Inspected pixel marker"),
        read_only=True,
    )
    n_bands = traitlets.Int(3, read_only=True, allow_none=False)

    def __init__(self, map, position="topright", layout=None):
        """
        Construct a PixelInspector and attach it to a map.

        Parameters
        ----------
        map: ipyleaflet.Map, dynamic_compute.interactive.MapApp
            The map to attach to
        position: str, optional, default "topright"
            Where on the map to display the values table
        layout: ipywidgets.Layout, optional
            Layout for the values table. Defaults to
            ``Layout(max_height="350px", overflow="scroll", padding="4px")``
        """
        if layout is None:
            layout = ipywidgets.Layout(
                max_height="350px", overflow="scroll", padding="4px"
            )
        super().__init__([], layout=layout)
        self.layout.grid_template_columns = "min-content " * (1 + self.n_bands)

        # Set up the lat/lon reader
        _value_layout = {"width": "initial", "margin": "0 2px", "height": "1.6em"}
        self.lat_label = ipywidgets.Label(value="Latitude,", layout=_value_layout)
        self.lat_label.layout.grid_column = "1"
        self.lon_label = ipywidgets.Label(value="Longitude:", layout=_value_layout)
        self.lon_label.layout.grid_column = "2"
        self.lat_val = ipywidgets.Label(value="0,", layout=_value_layout)
        self.lat_val.layout.grid_column = "3"
        self.lon_val = ipywidgets.Label(value="0", layout=_value_layout)
        self.lon_val.layout.grid_column = "4"

        # awkwardly handle MapApp without circularly importing it for an isinstance check
        try:
            sub_map = map.map
        except AttributeError:
            pass
        else:
            if isinstance(sub_map, ipyleaflet.Map):
                map = sub_map

        self._map = map
        self.marker.map = map
        self._inspector_rows_by_layer_id = weakref.WeakValueDictionary()

        self._layers_changed({"old": [], "new": map.layers})
        # initialize with the current layers on the map, if any

        self._control = ipyleaflet.WidgetControl(widget=self, position=position)
        map.add_control(self._control)
        map.observe(self._layers_changed, names=["layers"], type="change")
        map.on_interaction(self._handle_click)

        self._orig_cursor = map.default_style.cursor
        map.default_style.cursor = "crosshair"

    def unlink(self):
        "Stop listening for click events or layer updates and remove the table from the map"
        self._map.on_interaction(self._handle_click, remove=True)
        self._map.unobserve(self._layers_changed, "layers", type="change")
        for inspector_row in tuple(self._inspector_rows_by_layer_id.values()):
            # ^ take a tuple first, since unlinking should remove all references to `inspector_row`,
            # which would then pop it from the dict, causing mutation of the dict while we iterate over it
            inspector_row.unlink()
        self._map.default_style.cursor = self._orig_cursor
        try:
            self._map.remove_control(self._control)
        except ipyleaflet.ControlException:
            pass
        try:
            self._map.remove_layer(self.marker)
        except ipyleaflet.LayerException:
            pass
        self.marker.opacity = 0  # be extra sure no more inspects will run
        self.children = []

    def _layers_changed(self, change):
        new_layers = change["new"]

        inspector_rows = []
        for layer in reversed(new_layers):
            if isinstance(layer, DynamicComputeLayer):
                try:
                    inspector_row = self._inspector_rows_by_layer_id[layer.model_id]
                except KeyError:
                    inspector_row = InspectorRowGenerator(
                        layer, self.marker, self.n_bands
                    )
                    self._inspector_rows_by_layer_id[layer.model_id] = inspector_row

                inspector_rows.append(inspector_row)

        new_children = []
        new_children.extend(
            [self.lat_label, self.lon_label, self.lat_val, self.lon_val]
        )
        for inspector_row in inspector_rows:
            new_children.append(inspector_row.name_label)
            new_children.extend(inspector_row.values_labels)

        self.children = new_children

    def _handle_click(self, **kwargs):
        with self._map.output_log:
            if kwargs.get("type") != "click":
                return
            try:
                lat, lon = kwargs["coordinates"]
            except KeyError:
                return

            self.marker.opacity = 1
            self.marker.location = (lat, lon)
            self.lat_val.value = f"{self.marker.location[0]:.5f},"
            self.lon_val.value = f"{self.marker.location[1]:.5f}"

            # in case it accidentally got deleted with `clear_layers`
            if self.marker not in self._map.layers:
                self._map.add_layer(self.marker)


class InspectorRowGenerator(traitlets.HasTraits):
    """
    Controller class that manages the name and pixel values widgets for one layer.

    Not a widget itself, but just exposes `name_label` and `value_labels`
    for the `PixelInspector` to add into its table.

    Listens for changes to the layer (XYZ object or parameters) or the marker (location),
    and updates the widgets in `value_labels` appropriately, by calling `inspect` in a
    separate thread to pull pixel values.
    """

    _value_layout = {"width": "initial", "margin": "0 2px", "height": "1.6em"}

    name_label = traitlets.Instance(
        ipywidgets.Label,
        kw={"layout": dict(_value_layout, grid_column="1")},
        read_only=True,
        allow_none=False,
    )
    values_labels = traitlets.List(read_only=True, allow_none=False)

    def __init__(self, layer, marker, n_bands):
        self.marker = marker
        self.layer = layer
        self._updating = False
        self._cache = cachetools.LRUCache(64)

        self.name_label.value = layer.name
        # TODO make names bold. it's frustratingly difficult (unsupported) with ipywidgets:
        # https://github.com/jupyter-widgets/ipywidgets/issues/577
        self._name_link = ipywidgets.jslink((layer, "name"), (self.name_label, "value"))
        self.set_trait(
            "values_labels",
            [
                ipywidgets.Label(
                    value="", layout=dict(self._value_layout, grid_column=str(2 + i))
                )
                for i in range(n_bands)
            ],
        )

        marker.observe(self.recalculate, "inspect_latlon", type="change")
        layer.observe(self.recalculate, ["image_value", "visible"], type="change")

        self._viz_links = [
            traitlets.dlink(
                (layer, "visible"),
                (label.layout, "display"),
                lambda v: "" if v else "none",
            )
            for label in [self.name_label] + self.values_labels
        ]

        if marker.opacity == 1:
            # there's already a point to sample; eagerly recalculate now
            self.recalculate()

    def unlink(self):
        # NOTE(gabe): the traitlets docs say name=All (the default) should work,
        # but careful reading of the source shows it's a no-op.
        # we must explicitly unobserve for exactly the names and types we observed for.
        self.marker.unobserve(self.recalculate, "inspect_latlon", type="change")
        self.layer.unobserve(
            self.recalculate, ["image_value", "visible"], type="change"
        )
        self._name_link.unlink()
        for viz_link in self._viz_links:
            viz_link.unlink()

    def recalculate(self, *args, **kwargs):
        if self._updating or not self.layer.visible or self.marker.opacity == 0:
            return

        # xy_3857 = self.marker.xy_3857
        latlon = self.marker.inspect_latlon
        # try to make a cache key from the marker location, layer ID, reduction, and parameters.
        # if the parameters are unhashable (probably because they contain grafts),
        # we'll consider it a cache miss and go fetch.
        try:
            params_key = frozenset(self.layer.parameters.to_dict().items())
        except TypeError:
            cache_key = None
        else:
            cache_key = (latlon, self.layer.layer_id, self.layer.reduction, params_key)

        if cache_key:
            try:
                value_list = self._cache[cache_key]
            except KeyError:
                value_list = None
        else:
            value_list = None

        image = self.layer.image_value
        if image is None:

            value_list = ["❓"]

        if value_list:
            self.set_values(value_list)
        else:
            self.set_updating()
            # NOTE(gabe, but still relevant per Mat): I don't trust traitlets or ipywidgets to be thread-safe,
            # so we pull all values out of traits here and pass them in to the thread directly
            thread = threading.Thread(
                target=self._fetch_and_set_thread,
                args=(
                    image,
                    latlon,
                    # ctx,
                    cache_key,
                    self.layer.layer_id,
                ),
                daemon=True,
            )
            thread.start()

    def _fetch_and_set_thread(self, image, latlon, cache_key, layer_id):

        try:
            lat, lon = latlon
            value_list = value_at(image, lat, lon, layer_id)
        except JobTimeoutError:
            value_list = ["⏱"]
        except Exception as ex:
            value_list = ["💥"]
            print(ex)
        else:
            if len(value_list) == 0:
                # empty Image
                value_list = [np.ma.masked]
            self._cache[cache_key] = value_list

        self.set_values(value_list)

    def set_values(self, new_values_list):
        for i, value in enumerate(new_values_list):
            if isinstance(value, str):
                pass
            elif value is np.ma.masked:
                new_values_list[i] = "∅"
            else:
                new_values_list[i] = "{:.6g}".format(value)

        for i, label in enumerate(self.values_labels):
            try:
                label.value = new_values_list[i]
            except IndexError:
                label.value = ""

        self._updating = False

    def set_updating(self):
        self._updating = True

        for i, label in enumerate(self.values_labels):
            if label.value != "" or i == 0:
                label.value = "..."
