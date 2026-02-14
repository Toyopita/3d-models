#!/usr/bin/env python3
"""和風行灯（あんどん）- 菱格子モダンデザイン v1
Japanese Modern Andon Light - Diamond Lattice

Size S: 150mm H x 100mm W x 100mm D
Pattern: 60° diamond lattice, 2.5mm line width
"""
import trimesh
import numpy as np
import manifold3d as m3d
import os

# === DIMENSIONS (mm) ===
W, D, H = 100.0, 100.0, 150.0
WALL_T = 2.0
BASE_T = 3.0
TOP_T = 2.0
TOP_HOLE = 60.0  # top opening for light/cable

# === LATTICE ===
LW = 2.5          # line width
S = 9.0           # lattice spacing parameter
HOLE_DV = 2*(S - LW)            # 13mm vertical diagonal
HOLE_DH = HOLE_DV / np.sqrt(3)  # ~7.51mm horizontal diagonal
PITCH_H = 2*S / np.sqrt(3)      # ~10.39mm horizontal period
PITCH_V = S                      # 9mm vertical period
OFFSET_H = S / np.sqrt(3)       # ~5.20mm odd row offset

# === MARGINS ===
M_TOP, M_BOT, M_SIDE = 8.0, 8.0, 5.0
Z_LO = BASE_T + M_BOT
Z_HI = H - TOP_T - M_TOP


def make_diamond_y(cx, cz, yc, ext):
    """Diamond prism along Y axis (front/back walls)"""
    hv, hh, hy = HOLE_DV/2, HOLE_DH/2, ext/2
    v = np.array([
        [cx, yc-hy, cz+hv], [cx+hh, yc-hy, cz],
        [cx, yc-hy, cz-hv], [cx-hh, yc-hy, cz],
        [cx, yc+hy, cz+hv], [cx+hh, yc+hy, cz],
        [cx, yc+hy, cz-hv], [cx-hh, yc+hy, cz],
    ])
    f = np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
        [2,3,7],[2,7,6],[3,0,4],[3,4,7],
    ])
    return trimesh.Trimesh(vertices=v, faces=f)


def make_diamond_x(cy, cz, xc, ext):
    """Diamond prism along X axis (left/right walls)"""
    hv, hh, hx = HOLE_DV/2, HOLE_DH/2, ext/2
    v = np.array([
        [xc-hx, cy, cz+hv], [xc-hx, cy+hh, cz],
        [xc-hx, cy, cz-hv], [xc-hx, cy-hh, cz],
        [xc+hx, cy, cz+hv], [xc+hx, cy+hh, cz],
        [xc+hx, cy, cz-hv], [xc+hx, cy-hh, cz],
    ])
    f = np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
        [2,3,7],[2,7,6],[3,0,4],[3,4,7],
    ])
    return trimesh.Trimesh(vertices=v, faces=f)


def get_positions(wall_w):
    """Generate centered diamond positions for a wall"""
    x_lo = M_SIDE + HOLE_DH/2
    x_hi = wall_w - M_SIDE - HOLE_DH/2
    z_lo = Z_LO + HOLE_DV/2
    z_hi = Z_HI - HOLE_DV/2

    avail_h = z_hi - z_lo
    n_rows = int(avail_h / PITCH_V) + 1
    z_start = z_lo + (avail_h - (n_rows-1) * PITCH_V) / 2

    pos = []
    for row in range(n_rows):
        z = z_start + row * PITCH_V
        if z < z_lo or z > z_hi:
            continue
        off = OFFSET_H if row % 2 else 0.0
        avail_w = x_hi - x_lo - off
        if avail_w <= 0:
            continue
        n_cols = int(avail_w / PITCH_H) + 1
        x_start = x_lo + off + (avail_w - (n_cols-1) * PITCH_H) / 2
        for col in range(n_cols):
            x = x_start + col * PITCH_H
            if x_lo <= x <= x_hi:
                pos.append((x, z))
    return pos


# === BUILD ===
print("=== 和風行灯 菱格子 v1 ===")
print(f"箱: {W}x{D}x{H}mm  壁厚: {WALL_T}mm")
print(f"菱穴: {HOLE_DV:.1f}x{HOLE_DH:.1f}mm  線幅: {LW}mm\n")

# 1. Hollow box with base and top plate
outer = trimesh.creation.box([W, D, H])
outer.apply_translation([W/2, D/2, H/2])

inner = trimesh.creation.box([W-2*WALL_T, D-2*WALL_T, H-BASE_T-TOP_T])
inner.apply_translation([W/2, D/2, BASE_T + (H-BASE_T-TOP_T)/2])

box = outer.difference(inner, engine='manifold')
print("OK 箱体生成")

# 2. Top opening
cyl = trimesh.creation.cylinder(radius=TOP_HOLE/2, height=TOP_T*3, sections=64)
cyl.apply_translation([W/2, D/2, H])
box = box.difference(cyl, engine='manifold')
print("OK 天板開口")

# 3. Diamond cutouts
ext = WALL_T + 0.4
diamonds = []

fb_pos = get_positions(W)
for x, z in fb_pos:
    diamonds.append(make_diamond_y(x, z, D - WALL_T/2, ext))  # front
    diamonds.append(make_diamond_y(x, z, WALL_T/2, ext))       # back

lr_pos = get_positions(D)
for y, z in lr_pos:
    diamonds.append(make_diamond_x(y, z, WALL_T/2, ext))       # left
    diamonds.append(make_diamond_x(y, z, W - WALL_T/2, ext))   # right

n = len(diamonds)
print(f"OK 菱格子 {n}個 ({n//4}個/面 x 4面)")

# Convert trimesh to manifold3d and use compose for batch subtraction
def trimesh_to_manifold(mesh):
    return m3d.Manifold(m3d.Mesh(
        vert_properties=np.array(mesh.vertices, dtype=np.float64),
        tri_verts=np.array(mesh.faces, dtype=np.uint32)
    ))

def manifold_to_trimesh(manifold):
    mesh = manifold.to_mesh()
    return trimesh.Trimesh(
        vertices=mesh.vert_properties[:, :3],
        faces=mesh.tri_verts
    )

# Compose all diamonds (fast: no Boolean needed for non-overlapping parts)
print("  manifold変換中...")
diamond_manifolds = [trimesh_to_manifold(d) for d in diamonds]
combined_m = m3d.Manifold.compose(diamond_manifolds)

box_m = trimesh_to_manifold(box)
result_m = box_m - combined_m
result = manifold_to_trimesh(result_m)
print("OK Boolean演算完了")

# 4. Verify & Export
print(f"\n--- 検証 ---")
print(f"Watertight: {result.is_watertight}")
print(f"頂点: {len(result.vertices)}, 面: {len(result.faces)}")
print(f"体積: {result.volume:.0f} mm3")

path = '/Users/minamitakeshi/3d_models/andon/和風行灯_菱格子_v1.stl'
result.export(path)
print(f"\n✅ {path}")
