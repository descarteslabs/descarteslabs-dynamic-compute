import ipywidgets as widgets
import traitlets

from .utils import wrap_num


class LonLatInput(widgets.Text):
    """
    Input for entering lon, lat as comma-separated string

    Link to ``model``, not ``value``! ``model`` is the 2-list of floats,
    ``value`` is the displayed string value.

    Use ``model_is_latlon`` to reverse the order between the ``model`` and the ``value``.
    """

    model = traitlets.List(
        traitlets.CFloat(), default_value=(0.0, 0.0), minlen=2, maxlen=2
    )
    model_is_latlon = traitlets.Bool(False)
    description = traitlets.Unicode("Lat, lon (WGS84):").tag(sync=True)
    continuous_update = traitlets.Bool(False).tag(sync=True)

    @traitlets.observe("value")
    def _sync_view_to_model(self, change):
        new = change["new"]
        values = [part.strip() for part in new.split(",")]
        values = [values[0], str(wrap_num(float(values[1])))]
        if self.model_is_latlon:
            values.reverse()
        self.model = values

    @traitlets.observe("model")
    def _sync_model_to_view(self, change):
        new = change["new"]
        new_list = [wrap_num(new[0]), new[1]]
        string = "{:.4f}, {:.4f}".format(
            # https://xkcd.com/2170/
            *(reversed(new_list) if self.model_is_latlon else new_list)
        )
        self.value = string
