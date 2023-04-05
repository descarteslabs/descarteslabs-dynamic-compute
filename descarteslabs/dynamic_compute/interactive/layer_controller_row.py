# -*- coding: utf-8 -*-
import ipywidgets as widgets
import traitlets

# from .layer import WorkflowsLayer
from ipyleaflet import TileLayer

from .map_ import Map

initial_width = widgets.Layout(width="initial")
scale_width = widgets.Layout(min_width="1.3em", max_width="4em", width="initial")
scale_red_border = widgets.Layout(
    min_width="1.3em", max_width="4em", width="initial", border="2px solid #a81f00"
)
button_layout = widgets.Layout(width="initial", overflow="visible")


class CText(widgets.Text):
    value = traitlets.CUnicode(help="String value", allow_none=True).tag(sync=True)


class LayerControllerRow(widgets.Box):
    """
    Generic class for controlling a single `Layer` on a single `Map`.

    Provides controls for order/deletion on the map.

    Attributes
    ----------
    map: Map
        The map on which `layer` is displayed.
    layer: Layer
        An object you can add to a map via `m.add_layer`
    """

    map = traitlets.Instance(Map)
    _widgets = traitlets.Dict()

    def __init__(self, layer, map):
        self.layer = layer
        self.map = map

        name = widgets.Text(
            value=layer.name,
            placeholder="Layer name",
            layout=widgets.Layout(min_width="4em", max_width="12em"),
        )
        widgets.jslink((name, "value"), (layer, "name"))
        self._widgets["name"] = name

        move_up = widgets.Button(
            description="↑", tooltip="Move layer up", layout=button_layout
        )
        move_up.on_click(self.move_up)
        self._widgets["move_up"] = move_up

        move_down = widgets.Button(
            description="↓", tooltip="Move layer down", layout=button_layout
        )
        move_down.on_click(self.move_down)
        self._widgets["move_down"] = move_down

        remove = widgets.Button(
            description="✖︎", tooltip="Remove layer", layout=button_layout
        )
        remove.on_click(self.remove)
        self._widgets["remove"] = remove

        super(LayerControllerRow, self).__init__(self._make_children())

        self.layout.overflow = "initial"

    def move_up(self, _):
        "``on_click`` handler to move ``self.layer`` up on ``self.map``"
        self.map.move_layer_up(self.layer)

    def move_down(self, _):
        "``on_click`` handler to move ``self.layer`` down on ``self.map``"
        self.map.move_layer_down(self.layer)

    def remove(self, _):
        "``on_click`` handler to remove ``self.layer`` from ``self.map``"
        self.map.remove_layer(self.layer)

    def _make_children(self):
        widgets = self._widgets
        children = [
            widgets["name"],
            widgets["move_up"],
            widgets["move_down"],
            widgets["remove"],
        ]

        return children


class TileLayerControllerRow(LayerControllerRow):
    """
    Widget for the controls of a single `ipyleaflet.TileLayer` on a single `Map`.

    Provides controls for visbility, and order/deletion on the map.

    Attributes
    ----------
    map: Map
        The map on which `layer` is displayed.
    layer: TileLayer
        The layer this widget is controlling.
    """

    map = traitlets.Instance(Map)
    layer = traitlets.Instance(TileLayer)

    def __init__(self, layer, map):
        self.layer = layer
        self.map = map

        visible = widgets.Checkbox(
            value=layer.visible, layout=initial_width, indent=False
        )
        widgets.jslink((visible, "value"), (layer, "visible"))
        self._widgets["visible"] = visible
        opacity = widgets.FloatSlider(
            value=layer.opacity,
            min=0,
            max=1,
            step=0.01,
            continuous_update=True,
            readout=False,
            layout=widgets.Layout(max_width="50px", min_width="20px"),
        )
        widgets.jslink((opacity, "value"), (layer, "opacity"))
        self._widgets["opacity"] = opacity

        super().__init__(layer, map)

    def _make_children(self):
        widgets = self._widgets
        children = [
            widgets["visible"],
            widgets["opacity"],
            widgets["name"],
            widgets["move_up"],
            widgets["move_down"],
            widgets["remove"],
        ]

        return children
