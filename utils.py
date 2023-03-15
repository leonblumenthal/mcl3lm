import numpy as np
import plotly.graph_objects as go


def add_coordinate_frame_origin(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    name: str = "",
    scale: float = 0.2,
    colors: tuple[str, str, str] = ("red", "green", "blue"),
    axis_names: tuple[str, str, str] = ("x", "y", "z"),
):
    for axis, color, axis_name in zip(R.T, colors, axis_names):
        fig.add_scatter3d(
            x=[t[0], t[0] + axis[0] * scale],
            y=[t[1], t[1] + axis[1] * scale],
            z=[t[2], t[2] + axis[2] * scale],
            mode="lines",
            line_color=color,
            name=f"{axis_name} {name}",
            showlegend=False,
        )
