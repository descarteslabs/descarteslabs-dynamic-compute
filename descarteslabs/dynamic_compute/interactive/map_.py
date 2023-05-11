import math
import textwrap
from typing import Tuple

import ipyleaflet
import IPython
import ipywidgets as widgets
import traitlets

from .clearable import ClearableOutput
from .inspector import PixelInspector
from .lonlat import LonLatInput
from .utils import tuple_move, wrap_num

EARTH_EQUATORIAL_RADIUS_WGS84_M = 6378137.0

app_layout = widgets.Layout(height="100%", padding="0 0 8px 0")
import descarteslabs as dl


class MapController(widgets.HBox):
    "Widget for controlling the center/zoom of a `Map` and toggling the pixel inspector."

    def __init__(self, map):
        lonlat = LonLatInput(
            model=map.center,
            layout=widgets.Layout(width="initial"),
            style={"description_width": "initial"},
        )

        widgets.link((map, "center"), (lonlat, "model"))

        zoom_label = widgets.Label(
            value="Zoom:", layout=widgets.Layout(width="initial")
        )

        zoom = widgets.BoundedIntText(
            value=map.zoom,
            layout=widgets.Layout(width="3em"),
            min=map.min_zoom,
            max=map.max_zoom,
            step=1,
        )
        # TODO Get the Pixel Inspector working and add it here
        widgets.link((map, "zoom"), (zoom, "value"))
        widgets.link((map, "min_zoom"), (zoom, "min"))
        widgets.link((map, "max_zoom"), (zoom, "max"))

        inspect = widgets.ToggleButton(
            value=map.inspecting_pixels,
            description="Pixel inspector",
            tooltip="Calculate pixel values on click",
            icon="crosshairs",
            layout=widgets.Layout(width="initial", overflow="visible"),
        )
        widgets.link((map, "inspecting_pixels"), (inspect, "value"))

        # TODO add inspect to the list below once it's actually working
        super(MapController, self).__init__(
            children=(lonlat, zoom_label, zoom, inspect)
        )

        self.layout.overflow = "hidden"
        self.layout.flex = "0 0 auto"
        self.layout.padding = "2px 0"


class MapApp(widgets.VBox):
    """
    Widget displaying a map, layers, and output logs in a nicer layout.

    Forwards attributes and methods to ``self.map``.

    Note: to change the size of the map when displayed inline in a notebook,
    set ``dc.map.map.layout.height = "1000px"``. (``dc.map.map`` is the actual
    map widget, ``dc.map`` is the container with the layer controls, which will
    resize accordingly.) Setting the height on just ``dc.map`` will also work,
    but if you use an output view, then the map won't resize itself to fit in within it.

    """

    _forward_attrs_to_map = {
        "center",
        "zoom_start",
        "zoom",
        "max_zoom",
        "min_zoom",
        "interpolation",
        "crs",
        # Specification of the basemap
        "basemap",
        "modisdate",
        # Interaction options
        "dragging",
        "touch_zoom",
        "scroll_wheel_zoom",
        "double_click_zoom",
        "box_zoom",
        "tap",
        "tap_tolerance",
        "world_copy_jump",
        "close_popup_on_click",
        "bounce_at_zoom_limits",
        "keyboard",
        "keyboard_pan_offset",
        "keyboard_zoom_offset",
        "inertia",
        "inertia_deceleration",
        "inertia_max_speed",
        "zoom_animation_threshold",
        "fullscreen",
        "zoom_control",
        "attribution_control",
        "south",
        "north",
        "east",
        "west",
        "layers",
        "controls",
        "bounds",
        "bounds_polygon",
        "pixel_bounds",
        # from subclass
        "output_log",
        "logs",
        "inspecting_pixels",
        # methods
        "move_layer",
        "move_layer_up",
        "move_layer_down",
        "add_layer",
        "add_control",
        "remove_layer",
        "remove_control",
        "substitute",
        "clear_controls",
        "clear_layers",
        "on_interaction",
        "geocontext",
    }
    control_tile_layers = traitlets.Bool(
        default_value=True, help="Show controls for `ipyleaflet.TileLayer`s"
    )
    control_other_layers = traitlets.Bool(
        default_value=False,
        help="Show generic controls for other ipyleaflet layer types",
    )

    def __init__(self, map=None, layer_controller_list=None, map_controller=None):
        if map is None:
            map = Map()
            map.add_control(ipyleaflet.FullScreenControl())
            map.add_control(ipyleaflet.ScaleControl(position="bottomleft"))

        if layer_controller_list is None:
            from .layer_controller import LayerControllerList

            layer_controller_list = LayerControllerList(map)
        if map_controller is None:
            map_controller = MapController(map)

        self.map = map
        self.controller_list = layer_controller_list
        widgets.link(
            (self, "control_tile_layers"),
            (layer_controller_list, "control_tile_layers"),
        )
        widgets.link(
            (self, "control_other_layers"),
            (layer_controller_list, "control_other_layers"),
        )
        self.map_controller = map_controller

        def on_clear():
            for layer in self.map.layers:
                try:
                    layer.forget_logs()
                except AttributeError:
                    pass

        self.clearable_logs = ClearableOutput(
            map.logs,
            on_clear=on_clear,
            layout=widgets.Layout(max_height="20rem", flex="0 0 auto"),
        )

        self.autoscale_outputs = widgets.VBox(
            [
                # TODO once we get DynamicComputeLayer layer working properly, they will go here
                # x.autoscale_progress
                # for x in reversed(self.map.layers)
                # if isinstance(x, DynamicComputeLayer)
            ],
            layout=widgets.Layout(flex="0 0 auto", max_height="16rem"),
        )

        super(MapApp, self).__init__(
            [
                map,
                self.clearable_logs,
                map.output_log,
                self.autoscale_outputs,
                map_controller,
                layer_controller_list,
            ],
            layout=app_layout,
        )

        map.observe(self._update_autoscale_progress, names=["layers"])

    def _update_autoscale_progress(self, change):
        self.autoscale_outputs.children = [
            # TODO once we get DynamicComputeLayer layer working properly, they will go here
            # x.autoscale_progress
            # for x in reversed(self.map.layers)
            # if isinstance(x, DynamicComputeLayer)
        ]

    def __getattr__(self, attr):
        if attr in self._forward_attrs_to_map:
            return getattr(self.__dict__["map"], attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, x):
        if attr in self._forward_attrs_to_map:
            return setattr(self.__dict__["map"], attr, x)
        else:
            return super(MapApp, self).__setattr__(attr, x)

    def __dir__(self):
        return super(MapApp, self).__dir__() + list(self._forward_attrs_to_map)

    def __repr__(self):
        msg = """
        `ipyleaflet` and/or `ipywidgets` Jupyter extensions are not installed! (or you're not in a Jupyter notebook.)
        To install for JupyterLab, run this in a cell:
            !jupyter labextension install jupyter-leaflet @jupyter-widgets/jupyterlab-manager
        To install for plain Jupyter Notebook, run this in a cell:
            !jupyter nbextension enable --py --sys-prefix ipyleaflet
        Then, restart the kernel and refresh the webpage.
        """
        return textwrap.dedent(msg)

    def _ipython_display_(self, **kwargs):
        """
        Called when `IPython.display.display` is called on the widget.

        Copied verbatim from
        https://github.com/jupyter-widgets/ipywidgets/blob/master/ipywidgets/widgets/widget.py#L709-L729,
        but with truncation 110-character repr truncation removed, so we can display a helpful message when necessary
        extensions aren't installed.
        """

        plaintext = repr(self)
        data = {"text/plain": plaintext}
        if self._view_name is not None:
            # The 'application/vnd.jupyter.widget-view+json' mimetype has not been registered yet.
            # See the registration process and naming convention at
            # http://tools.ietf.org/html/rfc6838
            # and the currently registered mimetypes at
            # http://www.iana.org/assignments/media-types/media-types.xhtml.
            data["application/vnd.jupyter.widget-view+json"] = {
                "version_major": 2,
                "version_minor": 0,
                "model_id": self._model_id,
            }
        IPython.display.display(data, raw=True)

        if self._view_name is not None:
            self._handle_displayed(**kwargs)

    def _handle_displayed(self, **kwargs):
        pass


class Map(ipyleaflet.Map):
    """
    Subclass of ``ipyleaflet.Map`` with Workflows defaults and extra helper methods.

    Attributes
    ----------
    output_log: ipywidgets.Output
        Widget where functions doing operations on this map (especially compute operations,
        like autoscaling or timeseries) can log their output.

    """

    center = traitlets.List(
        [35.6870, -105.93780], help="Initial geographic center of the map"
    ).tag(sync=True, o=True)
    zoom_start = traitlets.Int(8, help="Initial map zoom level").tag(sync=True, o=True)
    zoom = traitlets.Int(8, help="Initial map map zoom level").tag(sync=True, o=True)
    min_zoom = traitlets.Int(5, help="Minimum allowable zoom level of the map").tag(
        sync=True, o=True
    )
    max_zoom = traitlets.Int(18, help="Maximum allowable zoom level of the map").tag(
        sync=True, o=True
    )

    scroll_wheel_zoom = traitlets.Bool(
        True, help="Whether the map can be zoomed by using the mouse wheel"
    ).tag(sync=True, o=True)

    logs = traitlets.Instance(
        widgets.Output,
        args=(),
        help="Widget where tiles layers can write their log messages.",
    )

    output_log = traitlets.Instance(
        widgets.Output,
        args=(),
        help="""
        Widget where functions doing operations on this map
        (especially compute operations) can log their output.
        """,
    )

    autoscale_outputs = traitlets.Instance(
        widgets.VBox,
        args=(),
        help="Widget containing all layers' autoscale output widgets",
    )
    inspecting_pixels = traitlets.Bool(
        False,
        help="Whether the pixel inspector is active, and clicking on the map displays pixel values",
    )

    def move_layer(self, layer, new_index):
        """
        Move a layer to a new index. Indices are one-indexed.

        Parameters
        ----------
        layer: ipyleaflet.Layer
            Leaflet layer object to move
        new_index: Int
            Index to move layer to. Indices are one-indexed.

        Raises
        ------
        ValueError
            If ``layer`` is a base layer, or does not already exist on the map.

        """
        if layer.base:
            raise ValueError("Cannot reorder base layer {}".format(layer))

        try:
            old_i = self.layers.index(layer)
        except ValueError:
            raise ValueError("Layer {} does not exist on the map".format(layer))

        self.layers = tuple_move(self.layers, old_i, new_index)

    def move_layer_up(self, layer):
        """
        Move a layer up one, if not already at the top.

        Parameters
        ----------
        layer: ipyleaflet.Layer

        Raises
        ------
        ValueError
            If ``layer`` is a base layer, or does not already exist on the map.

        """
        if layer.base:
            raise ValueError("Cannot reorder base layer {}".format(layer))

        try:
            old_i = self.layers.index(layer)
        except ValueError:
            raise ValueError("Layer {} does not exist on the map".format(layer))

        if old_i < len(self.layers) - 1:
            self.layers = tuple_move(self.layers, old_i, old_i + 1)

    def move_layer_down(self, layer):
        """
        Move a layer down one, if not already at the bottom.

        Parameters
        ----------
        layer: ipyleaflet.Layer

        Raises
        ------
        ValueError
            If ``layer`` is a base layer, or does not already exist on the map.

        """
        if layer.base:
            raise ValueError("Cannot reorder base layer {}".format(layer))

        try:
            old_i = self.layers.index(layer)
        except ValueError:
            raise ValueError("Layer {} does not exist on the map".format(layer))

        if old_i > 0 and not self.layers[old_i - 1].base:
            self.layers = tuple_move(self.layers, old_i, old_i - 1)

    def remove_layer(self, layer_name):
        """
        Remove a named layer or layer instance from the map

        Parameters
        ----------
        layer_name: Str or ipyleaflet.Layer
            Name of the layer or Layer instance to remove

        """
        if isinstance(layer_name, ipyleaflet.Layer):
            super().remove_layer(layer_name)
        else:
            for lyr in self.layers:
                if lyr.name == layer_name:
                    super().remove_layer(lyr)
                    break
            else:
                raise ValueError(
                    "Layer {} does not exist on the map".format(layer_name)
                )

    def clear_layers(self):
        """
        Remove all layers from the map (besides the base layer)

        """
        self.layers = tuple(lyr for lyr in self.layers if lyr.base)

    def map_dimensions(self) -> Tuple[float]:
        """
        Width, height of this Map, in pixels.

        """
        # https://github.com/jupyter-widgets/ipyleaflet/pull/616#issue-433563400
        (left, top), (right, bottom) = self.pixel_bounds
        return (math.ceil(bottom - top), math.ceil(right - left))

    def geocontext(self, resolution=None, shape=None, crs="EPSG:3857") -> dl.geo.AOI:
        """
        A Scenes :class:`~descarteslabs.geo.AOI` representing
        the current view area and resolution of the map. The ``bounds`` of the
        of the returned geocontext are the current valid bounds of the map viewport.

        Parameters
        ----------
        crs: str, default "EPSG:3857"
            Coordinate reference system into which data will be projected,
            expressed as an EPSG code (like ``EPSG:4326``), PROJ.4 definition,
            or ``"utm"``. If crs is ``"utm"``, the zone is calculated automatically
            from lat, lng of map center. Defaults to the Web Mercator projection
            (``EPSG:3857``).

        resolution: float, default: None
            Distance, in units of the ``crs``, that the edge of each pixel
            represents on the ground. Only one of ``resolution`` or ``shape``
            can be given. If neither ``shape`` nor ``resolution`` is given,
            ``shape`` defaults to the current dimensions of the map viewport.

        shape: tuple, default: None
            The dimensions (rows, columns), in pixels, to fit the output array within.
            Only one of ``resolution`` or ``shape`` can be given. If neither ``shape``
            nor ``resolution`` is given, ``shape`` defaults to the current dimensions
            of the map viewport.

        Returns
        -------
        geoctx: descarteslabs.geo.AOI
        """

        bounds = self.valid_4326_bounds()
        if bounds == [0, 0, 0, 0]:
            raise RuntimeError(
                "Undefined bounds, please ensure that the interactive map has "
                "finished rendering and run this cell again. If you have not "
                "displayed the map yet, run `dc.map` in its own cell first."
            )

        if shape is not None and resolution is not None:
            raise RuntimeError("Must set only one of `resolution` or `shape`")
        elif shape is None and resolution is None:
            shape = self.map_dimensions()

        if crs.lower() == "utm":
            lat, lng = self.center
            lng = wrap_num(lng)
            # Source: https://gis.stackexchange.com/questions/13291/computing-utm-zone-from-lat-long-point
            if 56.0 <= lat < 64.0 and 3.0 <= lng < 12.0:
                # western coast of Norway
                zone = 32
            elif lat >= 72.0 and lat < 84.0:
                # special zones for Svalbard
                if lng >= 0.0 and lng < 9.0:
                    zone = 31
                elif lng >= 9.0 and lng < 21.0:
                    zone = 33
                elif lng >= 21.0 and lng < 33.0:
                    zone = 35
                elif lng >= 33.0 and lng < 42.0:
                    zone = 37
            else:
                zone = math.floor((lng + 180) / 6) + 1

            crs = "+proj=utm +zone={} +datum=WGS84 +units=m +no_defs ".format(zone)

        return dl.geo.AOI(
            bounds=bounds,
            crs=crs,
            bounds_crs="EPSG:4326",
            resolution=resolution,
            shape=shape,
            align_pixels=False,
        )

    def valid_4326_bounds(self):
        """
        Returns a valid bounding box for crs 4326 in the case that someone has panned beyond the allowable extent.

        Returns
        -------
            list of coordinates [west, south, east, north] representing a bounding box
        """
        dl
        # TODO: This isn't quite right for envelopes that intersect the antimeridian - the dl.geo.AOI call will still
        # error on those because there's not a way to make a single bounding box rectangle that crosses the
        # antimeridian (it really should be two bboxes). (explanation https://www.rfc-editor.org/rfc/rfc7946#section-5.2).

        return [wrap_num(self.west), self.south, wrap_num(self.east), self.north]

    @traitlets.observe("inspecting_pixels", type="change")
    def _update_inspecting_pixels(self, change):
        current_inspector = getattr(self, "_inspector", None)

        if change["new"] is True:
            if current_inspector:
                return
            self._inspector = PixelInspector(self)
        else:
            if current_inspector:
                current_inspector.unlink()
                self._inspector = None
