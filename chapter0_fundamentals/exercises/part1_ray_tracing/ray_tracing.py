# %%

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
from plotly_utils import imshow

MAIN = __name__ == "__main__"


# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    num_points = 2
    num_dim = 3
    rays = t.zeros(num_pixels, num_points, num_dim, dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    print(rays)
    return rays


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%
cat = t.cat([t.ones(2, 2), t.zeros(2, 2)], dim=0)
cat.shape

stack = t.stack([t.ones(2, 2), t.zeros(2, 2)], dim=0)
stack.shape

# %%
# x +2y = 4
# 3x - 5y = 1
A = t.tensor([[1, 2], [3, -5]], dtype=t.float32)
B = t.tensor([4, 1], dtype=t.float32)
x = t.linalg.solve(A, B)
print(x)


# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    # We remove the 3rd dimension from both our ray and segment matrices as we don't currently use it
    ray = ray[:, :2]
    segment = segment[:, :2]
    print(f"Ray {ray}")
    print(f"Segment {segment}")

    # Ray = O + uD
    # Object = L_1 + v(L_2 - L_1)
    o, d = ray
    l1, l2 = segment

    # linalg.solve solves simulataneous equations
    # It calculates x for Ax = B where A is a matrix of our coefficients
    # and B is a vector of our ordinates
    # We can rearrange the equation to the form Ax = B
    # The intersection is when ray = segment so
    # O + uD = L_1 + v(L_2 - L_1)
    # Rearranging so our known values / ordinates are on the RHS gives us: 
    # uD - v(L_2 - L_1) = L_1 - O

    # Coeffs = D, (L_2 - L_1); Ordinates = L_1 - O
    coeffs: Float[Tensor, "points dims"] = t.stack([d, -(l2 - l1)], dim=-1)
    print(f"Coeffs {coeffs}")
    ordinates = l1 - o
    print(f"Ordinates {ordinates}")
    try:
        intersection: Float[Tensor, "points dims"] = t.linalg.solve(coeffs, ordinates)
    except RuntimeError as e:
        print(f"Error: {e}")
        return False
    print(f"Intersection {intersection}")
    # Validate the intersection, we cannot go beyond the origin starting point so u > 0
    # The segment has starting and end points so 0 <= v <= 1
    u = intersection[0].item()
    v = intersection[1].item()
    return u >= 0 and 0 <= v <= 1


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    num_rays = rays.shape[0]
    num_segments = segments.shape[0]
    rays2d: Float[Tensor, "nrays 2 2"] = rays[...,:2]
    segments2d: Float[Tensor, "nsegments 2 2"] = segments[...,:2]
    ray_repeats = einops.repeat(rays2d, "nrays p1 p2 -> nrays nsegments p1 p2", nsegments=num_segments)
    seg_repeats = einops.repeat(segments2d, "nsegments p1 p2 -> nrays nsegments p1 p2", nrays=num_rays)
    o = ray_repeats[:, :, 0]
    d = ray_repeats[:, :, 1]
    l1 = seg_repeats[:, :, 0]
    l2 = seg_repeats[:, :, 1]
    assert d.shape == (num_rays, num_segments, 2), f"Shape mismatch: got {d.shape}, expected {(num_rays, num_segments, 2)}"
    assert l1.shape == (num_rays, num_segments, 2), f"Shape mismatch: got {l1.shape}, expected {(num_rays, num_segments, 2)}"

    coeffs = t.stack([d, l1 - l2], dim=-1)
    assert coeffs.shape == (num_rays, num_segments, 2, 2)
    determinates: Float[Tensor, "num_rays num_segments"] = t.linalg.det(coeffs)
    is_singular: Bool[Tensor, "num_rays num_segments"] = determinates.abs() < 1e-8
    assert is_singular.shape == (num_rays, num_segments), f"Shape mismatch: got {is_singular.shape}, expected {(num_rays, num_segments)}"
    coeffs[is_singular] = t.eye(2)

    ordinates = l1 - o
    intersections: Float[Tensor, "rays segments 2"] = t.linalg.solve(coeffs, ordinates)
    u: Float[Tensor, "rays segments"] = intersections[...,0]
    v: Float[Tensor, "rays segments"] = intersections[...,1]
    has_intersection: Bool[Tensor, "rays segments"] = (u >= 0) & (v >= 0) & (v <= 1) & ~is_singular
    
    return has_intersection.any(dim=-1)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
x = t.tensor([[1,2,3], [4,5,6], [7,8,9]])
rep1 = einops.repeat(x, "x y -> 2 x y")
rep2 = einops.repeat(x, "x y -> x 2 y")
rep3 = einops.repeat(x, "x y -> x y 2")

# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    num_rays = num_pixels_y * num_pixels_z
    rays: Float[Tensor, "rays points 3"] = t.zeros(num_rays, 2, 3, dtype=t.float)
    y_values = t.linspace(start=-y_limit, end=y_limit, steps=num_pixels_y)
    z_values = t.linspace(start=-z_limit, end=z_limit, steps=num_pixels_z)
    rays[:,1,1] = einops.repeat(y_values, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(z_values, "z -> (y z)", y=num_pixels_y)
    rays[:, 1, 0] = 1
    assert rays.shape == (num_rays, 2, 3)
    return rays


def make_rays_2d_alt(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    This creates the same rays but does it dimension by dimension using t.stack

    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    num_rays = num_pixels_y * num_pixels_z
    x = t.ones(num_pixels_y * num_pixels_z, dtype=t.float)
    y = t.linspace(start=-y_limit, end=y_limit, steps=num_pixels_y, dtype=t.float)
    y = einops.repeat(y, "y -> (y z)", z =num_pixels_z)
    z = t.linspace(start=-z_limit, end=z_limit, steps=num_pixels_z, dtype=t.float)
    z = einops.repeat(z, "z -> (y z)", y=num_pixels_y)
    d = t.stack([x,y,z], dim=-1)
    o = t.zeros(num_pixels_y * num_pixels_z, 3, dtype=t.float)
    rays = t.stack([o,d], dim=1)
    print(rays.shape)
    assert rays.shape == (num_rays, 2, 3)
    return rays


# num_pixels_y = 10; num_pixels_z = 10
# y_limit, z_limit = (0.3, 0.3)

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%
one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
display(fig)


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)
# %%
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    coeffs: Float[Tensor, "coeffs 3"] = t.stack([-D, B - A, C - A], dim=-1)
    coeffs.shape
    ordinates = O - A
    try:
        intersects = t.linalg.solve(coeffs, ordinates)
    except:
        return False
    s, u, v = intersects
    s = s.item(); u = u.item(); v = v.item()
    return u >= 0 and v >= 0 and (u + v) <= 1 and s >= 0

A: Point = t.tensor([1,0,0], dtype=t.float); B: Point = t.tensor([1,1,-1], dtype=t.float); C: Point = t.tensor([1,1,1], dtype=t.float) 
O: Point = t.tensor([1,0,0], dtype=t.float); D: Point = t.tensor([1,1,0], dtype=t.float)

coeffs: Float[Tensor, "coeffs 3"] = t.stack([-D, B - A, C - A], dim=1)
coeffs.shape
ordinates = O - A
try:
    intersects = t.linalg.solve(coeffs, ordinates)
except:
    print("No Intersection")
    # return False
s, u, v = intersects
s = s.item(); u = u.item(); v = v.item()
print(u >= 0 and v >= 0 and (u + v) <= 1 and s >= 0)

tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    num_rays = rays.size(dim=0)
    o: Float[Tensor, "rays 3"] = rays[:, 0, :]
    d: Float[Tensor, "rays 3"] = rays[:, 1, :]
    triangles = einops.repeat(triangle, "points xyz -> rays points xyz", rays=num_rays)
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    coeffs_list: Float[Tensor, "rays coeffs 3"] = t.stack([-d, b - a, c - a], dim=-1)
    determinates: Float[Tensor, "rays"] = coeffs_list.det()
    is_singular: Bool[Tensor, "rays"] = determinates.abs() < 1e-8
    print(dets.shape)
    print(mat[is_singular].shape)
    print(t.eye(3).shape)
    coeffs_list[is_singular].dtype
    t.eye(3).dtype
    coeffs_list[is_singular] = t.eye(3)

    ordinates_list = o - a
    intersections: Float[Tensor, "rays 3"] = t.linalg.solve(coeffs_list, ordinates_list)
    s_list, u_list, v_list = intersections.unbind(1)
    result: Bool[Tensor, "rays"] = (s_list >= 0) & (u_list >= 0) & (v_list >= 0) & ((u_list + v_list) <= 1) & ~is_singular
    return result




A = t.tensor([1, 0.0, -0.5], dtype=t.float)
B = t.tensor([1, -0.5, 0.0], dtype=t.float)
C = t.tensor([1, 0.5, 0.5], dtype=t.float)
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)
rays = rays2d; triangle = test_triangle

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%

def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> NR pts dims", NR=NR).unbind(dim=1)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], dim=-1)
    
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    num_rays = rays.size(0); num_triangles = triangles.size(0)
    o, d = rays.unbind(1)
    assert d.shape == (num_rays, 3)
    o = einops.repeat(o, "rays xyz -> rays triangles xyz", triangles=num_triangles)
    d = einops.repeat(d, "rays xyz -> rays triangles xyz", triangles=num_triangles)
    a, b, c = triangles.unbind(1)
    a = einops.repeat(a, "triangles xyz -> rays triangles xyz", rays=num_rays)
    a, b, c = (einops.repeat(x, "triangles xyz -> rays triangles xyz", rays=num_rays) for x in triangles.unbind(1))
    assert a.shape == (num_rays, num_triangles, 3)

    coeffs: Float[Tensor, "rays triangles coeffs xyz"] = t.stack([-d, b - a, c - a], dim=-1)
    is_singular: Bool[Tensor, "rays triangles coeffs"] = t.linalg.det(coeffs).abs() < 1e-8
    coeffs[is_singular] = t.eye(3)
    assert coeffs.shape == (num_rays, num_triangles, 3, 3)
    ordinates: Float[Tensor, "rays triangles xyz"] = o - a

    intersections: Float[Tensor, "rays triangles suv"] = t.linalg.solve(coeffs, ordinates)
    assert intersections.shape == (num_rays, num_triangles, 3)
    s, u, v = intersections.unbind(-1)
    is_valid = ~is_singular & (u >=0) & (v >=0) & ((u + v) <= 1)
    s[~is_valid] = t.inf
    return einops.reduce(s, "rays triangles -> rays", reduction="min")

triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])

dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()