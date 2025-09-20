"""
Colormap
--------

Utility module for dealing with colormaps.

"""

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

from jinja2 import Template

from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler

rootpath: str = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(rootpath, "_cnames.json")) as f:
    _cnames: Dict[str, str] = json.loads(f.read())

with open(os.path.join(rootpath, "_schemes.json")) as f:
    _schemes: Dict[str, List[str]] = json.loads(f.read())


TypeRGBInts = Tuple[int, int, int]
TypeRGBFloats = Tuple[float, float, float]
TypeRGBAInts = Tuple[int, int, int, int]
TypeRGBAFloats = Tuple[float, float, float, float]
TypeAnyColorType = Union[TypeRGBInts, TypeRGBFloats, TypeRGBAInts, TypeRGBAFloats, str]


def _is_hex(x: str) -> bool:
    return x.startswith("#") and len(x) == 7


def _parse_hex(color_code: str) -> TypeRGBAFloats:
    return (
        _color_int_to_float(int(color_code[1:3], 16)),
        _color_int_to_float(int(color_code[3:5], 16)),
        _color_int_to_float(int(color_code[5:7], 16)),
        1.0,
    )


def _color_int_to_float(x: int) -> float:
    """Convert an integer between 0 and 255 to a float between 0. and 1.0"""
    return x / 255.0


def _color_float_to_int(x: float) -> int:
    """Convert a float between 0. and 1.0 to an integer between 0 and 255"""
    return int(x * 255.9999)


def _parse_color(x: Union[tuple, list, str]) -> TypeRGBAFloats:
    if isinstance(x, (tuple, list)):
        return tuple(tuple(x) + (1.0,))[:4]  # type: ignore
    elif isinstance(x, str) and _is_hex(x):
        return _parse_hex(x)
    elif isinstance(x, str):
        cname = _cnames.get(x.lower(), None)
        if cname is None:
            raise ValueError(f"Unknown color {cname!r}.")
        return _parse_hex(cname)
    else:
        raise ValueError(f"Unrecognized color code {x!r}")


def _base(x: float) -> float:
    if x > 0:
        base = pow(10, math.floor(math.log10(x)))
        return round(x / base) * base
    else:
        return 0


class ColorMap(MacroElement):
    """A generic class for creating colormaps.

    Parameters
    ----------
    vmin: float
        The left bound of the color scale.
    vmax: float
        The right bound of the color scale.
    caption: str
        A caption to draw with the colormap.
    text_color: str, default "black"
        The color for the text.
    max_labels : int, default 10
        Maximum number of legend tick labels
    """

    _template: Template = ENV.get_template("color_scale.js")

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        caption: str = "",
        text_color: str = "black",
        max_labels: int = 10,
    ):
        super().__init__()
        self._name = "ColorMap"

        self.vmin = vmin
        self.vmax = vmax
        self.caption = caption
        self.text_color = text_color
        self.index: List[float] = [vmin, vmax]
        self.max_labels = max_labels
        self.tick_labels: Optional[Sequence[Union[float, str]]] = None

        self.width = 450
        self.height = 40

    def render(self, **kwargs):
        """Renders the HTML representation of the element."""
        self.color_domain = [
            float(self.vmin + (self.vmax - self.vmin) * k / 499.0) for k in range(500)
        ]
        self.color_range = [self.__call__(x) for x in self.color_domain]

        # sanitize possible numpy floats to native python floats
        self.index = [float(i) for i in self.index]

        if self.tick_labels is None:
            self.tick_labels = legend_scaler(self.index, self.max_labels)

        super().render(**kwargs)

        figure = self.get_root()
        assert isinstance(figure, Figure), (
            "You cannot render this Element " "if it is not in a Figure."
        )

        figure.header.add_child(
            JavascriptLink("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"),
            name="d3",
        )  # noqa

    def rgba_floats_tuple(self, x: float) -> TypeRGBAFloats:
        """
        This class has to be implemented for each class inheriting from
        Colormap. This has to be a function of the form float ->
        (float, float, float, float) describing for each input float x,
        the output color in RGBA format;
        Each output value being between 0 and 1.
        """
        raise NotImplementedError

    def rgba_bytes_tuple(self, x: float) -> TypeRGBAInts:
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
        return tuple(_color_float_to_int(u) for u in self.rgba_floats_tuple(x))  # type: ignore

    def rgb_bytes_tuple(self, x: float) -> TypeRGBInts:
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B) with int values between 0 and 255.
        """
        return self.rgba_bytes_tuple(x)[:3]

    def rgb_hex_str(self, x: float) -> str:
        """Provides the color corresponding to value `x` in the
        form of a string of hexadecimal values "#RRGGBB".
        """
        return "#%02x%02x%02x" % self.rgb_bytes_tuple(x)

    def rgba_hex_str(self, x: float) -> str:
        """Provides the color corresponding to value `x` in the
        form of a string of hexadecimal values "#RRGGBBAA".
        """
        return "#%02x%02x%02x%02x" % self.rgba_bytes_tuple(x)

    def __call__(self, x: float) -> str:
        """Provides the color corresponding to value `x` in the
        form of a string of hexadecimal values "#RRGGBBAA".
        """
        return self.rgba_hex_str(x)

    def _repr_html_(self) -> str:
        """Display the colormap in a Jupyter Notebook.

        Does not support all the class arguments.

        """
        nb_ticks = 7
        delta_x = math.floor(self.width / (nb_ticks - 1))
        x_ticks = [(i) * delta_x for i in range(0, nb_ticks)]
        delta_val = delta_x * (self.vmax - self.vmin) / self.width
        val_ticks = [round(self.vmin + (i) * delta_val, 1) for i in range(0, nb_ticks)]

        return (
            f'<svg height="40" width="{self.width}">'
            + "".join(
                [
                    (
                        '<line x1="{i}" y1="15" x2="{i}" '
                        'y2="27" style="stroke:{color};stroke-width:2;" />'
                    ).format(
                        i=i * 1,
                        color=self.rgba_hex_str(
                            self.vmin + (self.vmax - self.vmin) * i / (self.width - 1),
                        ),
                    )
                    for i in range(self.width)
                ],
            )
            + (
                '<text x="0" y="38" style="text-anchor:start; font-size:11px;'
                ' font:Arial; fill:{}">{}</text>'
            ).format(
                self.text_color,
                self.vmin,
            )
            + "".join(
                [
                    (
                        '<text x="{}" y="38"; style="text-anchor:middle; font-size:11px;'
                        ' font:Arial; fill:{}">{}</text>'
                    ).format(x_ticks[i], self.text_color, val_ticks[i])
                    for i in range(1, nb_ticks - 1)
                ],
            )
            + (
                '<text x="{}" y="38" style="text-anchor:end; font-size:11px;'
                ' font:Arial; fill:{}">{}</text>'
            ).format(
                self.width,
                self.text_color,
                self.vmax,
            )
            + '<text x="0" y="12" style="font-size:11px; font:Arial; fill:{}">{}</text>'.format(
                self.text_color,
                self.caption,
            )
            + "</svg>"
        )


class LinearColormap(ColorMap):
    """Creates a ColorMap based on linear interpolation of a set of colors
    over a given index.

    Parameters
    ----------

    colors : list-like object with at least two colors.
        The set of colors to be used for interpolation.
        Colors can be provided in the form:
        * tuples of RGBA ints between 0 and 255 (e.g: `(255, 255, 0)` or
        `(255, 255, 0, 255)`)
        * tuples of RGBA floats between 0. and 1. (e.g: `(1.,1.,0.)` or
        `(1., 1., 0., 1.)`)
        * HTML-like string (e.g: `"#ffff00`)
        * a color name or shortcut (e.g: `"y"` or `"yellow"`)
    index : list of floats, default None
        The values corresponding to each color.
        It has to be sorted, and have the same length as `colors`.
        If None, a regular grid between `vmin` and `vmax` is created.
    vmin : float, default 0.
        The minimal value for the colormap.
        Values lower than `vmin` will be bound directly to `colors[0]`.
    vmax : float, default 1.
        The maximal value for the colormap.
        Values higher than `vmax` will be bound directly to `colors[-1]`.
    caption: str
        A caption to draw with the colormap.
    text_color: str, default "black"
        The color for the text.
    max_labels : int, default 10
        Maximum number of legend tick labels
    tick_labels: list of floats, default None
        If given, used as the positions of ticks."""

    def __init__(
        self,
        colors: Sequence[TypeAnyColorType],
        index: Optional[Sequence[float]] = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
        caption: str = "",
        text_color: str = "black",
        max_labels: int = 10,
        tick_labels: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            caption=caption,
            text_color=text_color,
            max_labels=max_labels,
        )
        self.tick_labels: Optional[Sequence[float]] = tick_labels

        n = len(colors)
        if n < 2:
            raise ValueError("You must provide at least 2 colors.")
        if index is None:
            self.index = [vmin + (vmax - vmin) * i * 1.0 / (n - 1) for i in range(n)]
        else:
            self.index = list(index)
        self.colors: List[TypeRGBAFloats] = [_parse_color(x) for x in colors]

    def rgba_floats_tuple(self, x: float) -> TypeRGBAFloats:
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with float values between 0. and 1.
        """
        if x <= self.index[0]:
            return self.colors[0]
        if x >= self.index[-1]:
            return self.colors[-1]

        i = len([u for u in self.index if u < x])  # 0 < i < n.
        if self.index[i - 1] < self.index[i]:
            p = (x - self.index[i - 1]) * 1.0 / (self.index[i] - self.index[i - 1])
        elif self.index[i - 1] == self.index[i]:
            p = 1.0
        else:
            raise ValueError("Thresholds are not sorted.")

        return tuple(  # type: ignore
            (1.0 - p) * self.colors[i - 1][j] + p * self.colors[i][j] for j in range(4)
        )

    def to_step(
        self,
        n: Optional[int] = None,
        index: Optional[Sequence[float]] = None,
        data: Optional[Sequence[float]] = None,
        method: str = "linear",
        quantiles: Optional[Sequence[float]] = None,
        round_method: Optional[str] = None,
        max_labels: int = 10,
    ) -> "StepColormap":
        """Splits the LinearColormap into a StepColormap.

        Parameters
        ----------
        n : int, default None
            The number of expected colors in the output StepColormap.
            This will be ignored if `index` is provided.
        index : list of floats, default None
            The values corresponding to each color bounds.
            It has to be sorted.
            If None, a regular grid between `vmin` and `vmax` is created.
        data : list of floats, default None
            A sample of data to adapt the color map to.
        method : str, default 'linear'
            The method used to create data-based colormap.
            It can be 'linear' for linear scale, 'log' for logarithmic,
            or 'quant' for data's quantile-based scale.
        quantiles : list of floats, default None
            Alternatively, you can provide explicitly the quantiles you
            want to use in the scale.
        round_method : str, default None
            The method used to round thresholds.
            * If 'int', all values will be rounded to the nearest integer.
            * If 'log10', all values will be rounded to the nearest
            order-of-magnitude integer. For example, 2100 is rounded to
            2000, 2790 to 3000.
        max_labels : int, default 10
            Maximum number of legend tick labels

        Returns
        -------
        A StepColormap with `n=len(index)-1` colors.

        Examples:
        >> lc.to_step(n=12)
        >> lc.to_step(index=[0, 2, 4, 6, 8, 10])
        >> lc.to_step(data=some_list, n=12)
        >> lc.to_step(data=some_list, n=12, method='linear')
        >> lc.to_step(data=some_list, n=12, method='log')
        >> lc.to_step(data=some_list, n=12, method='quantiles')
        >> lc.to_step(data=some_list, quantiles=[0, 0.3, 0.7, 1])
        >> lc.to_step(data=some_list, quantiles=[0, 0.3, 0.7, 1],
        ...           round_method='log10')

        """
        msg = "You must specify either `index` or `n`"
        if index is None:
            if data is None:
                if n is None:
                    raise ValueError(msg)
                else:
                    index = [
                        self.vmin + (self.vmax - self.vmin) * i * 1.0 / n
                        for i in range(1 + n)
                    ]
                    scaled_cm = self
            else:
                max_ = max(data)
                min_ = min(data)
                scaled_cm = self.scale(vmin=min_, vmax=max_)
                method = "quantiles" if quantiles is not None else method
                if method.lower().startswith("lin"):
                    if n is None:
                        raise ValueError(msg)
                    index = [min_ + i * (max_ - min_) * 1.0 / n for i in range(1 + n)]
                elif method.lower().startswith("log"):
                    if n is None:
                        raise ValueError(msg)
                    if min_ <= 0:
                        msg = "Log-scale works only with strictly " "positive values."
                        raise ValueError(msg)
                    index = [
                        math.exp(
                            math.log(min_)
                            + i * (math.log(max_) - math.log(min_)) * 1.0 / n,
                        )
                        for i in range(1 + n)
                    ]
                elif method.lower().startswith("quant"):
                    if quantiles is None:
                        if n is None:
                            msg = (
                                "You must specify either `index`, `n` or" "`quantiles`."
                            )
                            raise ValueError(msg)
                        else:
                            quantiles = [i * 1.0 / n for i in range(1 + n)]
                    p = len(data) - 1
                    s = sorted(data)
                    index = [
                        s[int(q * p)] * (1.0 - (q * p) % 1)
                        + s[min(int(q * p) + 1, p)] * ((q * p) % 1)
                        for q in quantiles
                    ]
                else:
                    raise ValueError(f"Unknown method {method}")
        else:
            scaled_cm = self.scale(vmin=min(index), vmax=max(index))

        n = len(index) - 1

        if round_method == "int":
            index = [round(x) for x in index]

        if round_method == "log10":
            index = [_base(x) for x in index]

        colors = [
            scaled_cm.rgba_floats_tuple(
                index[i] * (1.0 - i / (n - 1.0)) + index[i + 1] * i / (n - 1.0),
            )
            for i in range(n)
        ]

        caption = self.caption
        text_color = self.text_color

        return StepColormap(
            colors,
            index=index,
            vmin=index[0],
            vmax=index[-1],
            caption=caption,
            text_color=text_color,
            max_labels=max_labels,
            tick_labels=self.tick_labels,
        )

    def scale(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        max_labels: int = 10,
    ) -> "LinearColormap":
        """Transforms the colorscale so that the minimal and maximal values
        fit the given parameters.
        """
        return LinearColormap(
            self.colors,
            index=[
                vmin + (vmax - vmin) * (x - self.vmin) * 1.0 / (self.vmax - self.vmin)
                for x in self.index
            ],  # noqa
            vmin=vmin,
            vmax=vmax,
            caption=self.caption,
            text_color=self.text_color,
            max_labels=max_labels,
        )


class StepColormap(ColorMap):
    """Creates a ColorMap based on linear interpolation of a set of colors
    over a given index.

    Parameters
    ----------
    colors : list-like object
        The set of colors to be used for interpolation.
        Colors can be provided in the form:
        * tuples of int between 0 and 255 (e.g: `(255,255,0)` or
        `(255, 255, 0, 255)`)
        * tuples of floats between 0. and 1. (e.g: `(1.,1.,0.)` or
        `(1., 1., 0., 1.)`)
        * HTML-like string (e.g: `"#ffff00`)
        * a color name or shortcut (e.g: `"y"` or `"yellow"`)
    index : list of floats, default None
        The bounds of the colors. The lower value is inclusive,
        the upper value is exclusive.
        It has to be sorted, and have the same length as `colors`.
        If None, a regular grid between `vmin` and `vmax` is created.
    vmin : float, default 0.
        The minimal value for the colormap.
        Values lower than `vmin` will be bound directly to `colors[0]`.
    vmax : float, default 1.
        The maximal value for the colormap.
        Values higher than `vmax` will be bound directly to `colors[-1]`.
    caption: str
        A caption to draw with the colormap.
    text_color: str, default "black"
        The color for the text.
    max_labels : int, default 10
        Maximum number of legend tick labels
    tick_labels: list of floats, default None
        If given, used as the positions of ticks.
    """

    def __init__(
        self,
        colors: Sequence[TypeAnyColorType],
        index: Optional[Sequence[float]] = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
        caption: str = "",
        text_color: str = "black",
        max_labels: int = 10,
        tick_labels: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            caption=caption,
            text_color=text_color,
            max_labels=max_labels,
        )
        self.tick_labels = tick_labels

        n = len(colors)
        if n < 1:
            raise ValueError("You must provide at least 1 colors.")
        if index is None:
            self.index = [vmin + (vmax - vmin) * i * 1.0 / n for i in range(n + 1)]
        else:
            self.index = list(index)
        self.colors: List[TypeRGBAFloats] = [_parse_color(x) for x in colors]

    def rgba_floats_tuple(self, x: float) -> TypeRGBAFloats:
        """
        Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with float values between 0. and 1.

        """
        if x <= self.index[0]:
            return self.colors[0]
        if x >= self.index[-1]:
            return self.colors[-1]

        i = len([u for u in self.index if u <= x])  # 0 < i < n.
        return self.colors[i - 1]

    def to_linear(
        self,
        index: Optional[Sequence[float]] = None,
        max_labels: int = 10,
    ) -> LinearColormap:
        """
        Transforms the StepColormap into a LinearColormap.

        Parameters
        ----------
        index : list of floats, default None
                The values corresponding to each color in the output colormap.
                It has to be sorted.
                If None, a regular grid between `vmin` and `vmax` is created.
        max_labels : int, default 10
            Maximum number of legend tick labels

        """
        if index is None:
            n = len(self.index) - 1
            index = [
                self.index[i] * (1.0 - i / (n - 1.0))
                + self.index[i + 1] * i / (n - 1.0)
                for i in range(n)
            ]

        colors = [self.rgba_floats_tuple(x) for x in index]
        return LinearColormap(
            colors,
            index=index,
            vmin=self.vmin,
            vmax=self.vmax,
            caption=self.caption,
            text_color=self.text_color,
            max_labels=max_labels,
        )

    def scale(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        max_labels: int = 10,
    ) -> "StepColormap":
        """Transforms the colorscale so that the minimal and maximal values
        fit the given parameters.
        """
        return StepColormap(
            self.colors,
            index=[
                vmin + (vmax - vmin) * (x - self.vmin) * 1.0 / (self.vmax - self.vmin)
                for x in self.index
            ],  # noqa
            vmin=vmin,
            vmax=vmax,
            caption=self.caption,
            text_color=self.text_color,
            max_labels=max_labels,
        )


class _LinearColormaps:
    """A class for hosting the list of built-in linear colormaps."""

    def __init__(self):
        self._schemes = _schemes.copy()
        self._colormaps = {key: LinearColormap(val) for key, val in _schemes.items()}
        for key, val in _schemes.items():
            setattr(self, key, LinearColormap(val))

    def _repr_html_(self) -> str:
        return Template(
            """
        <table>
        {% for key,val in this._colormaps.items() %}
        <tr><td>{{key}}</td><td>{{val._repr_html_()}}</td></tr>
        {% endfor %}</table>
        """,
        ).render(this=self)


linear = _LinearColormaps()


class _StepColormaps:
    """A class for hosting the list of built-in step colormaps."""

    def __init__(self):
        self._schemes = _schemes.copy()
        self._colormaps = {key: StepColormap(val) for key, val in _schemes.items()}
        for key, val in _schemes.items():
            setattr(self, key, StepColormap(val))

    def _repr_html_(self) -> str:
        return Template(
            """
        <table>
        {% for key,val in this._colormaps.items() %}
        <tr><td>{{key}}</td><td>{{val._repr_html_()}}</td></tr>
        {% endfor %}</table>
        """,
        ).render(this=self)


step = _StepColormaps()
