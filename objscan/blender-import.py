# === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
#
# Purpose: imports the output of objscan-test into Blender 2.90 as a point cloud
#

# Turns the input file into a list of 3D vertices.
def parse_pointlist(path):
    parser = lambda line: [float(x) for x in line.rstrip().split()]
    with open(path, "r") as f:
        vertices = [parser(line) for line in f]
        return vertices

# Creates the object containing all the points.
def create_point_cloud(path):
    vertices = parse_pointlist(path)
    mesh = bpy.data.meshes.new(path + "_mesh")
    obj = bpy.data.objects.new(path + "_obj", mesh)
    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh = bpy.context.object.data
    bm = bmesh.new()
    for vtx in vertices:
        bm.verts.new(vtx)
    bm.to_mesh(mesh)
    bm.free()
