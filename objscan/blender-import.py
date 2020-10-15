# === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
#
# Purpose: imports the output of objscan-test into Blender 2.90 as a point cloud
#

# Turns the input file into a list of 3D vertices.
def parse_pointlist(path):
    f_split = lambda line: [x for x in line.rstrip().split()]
    parser = lambda line: [float(x) for x in line.rstrip().split()]
    f_parse_vertices = lambda arr: [float(x) for x in arr[1:]]
    f_parse_connection = lambda arr: [int(x) for x in arr[1:]]
    f_parser = { "v" : f_parse_vertices, "c" : f_parse_connection }
    ret = {}
    ret["v"] = []
    ret["c"] = []
    with open(path, "r") as f:
        arrays = [f_split(line) for line in f]
        for arr in arrays:
            key = arr[0]
            ret[key].append(f_parser[key](arr))
    return ret

# Creates the object containing all the points.
def create_point_cloud(path):
    sim = parse_pointlist(path)
    mesh = bpy.data.meshes.new(path + "_mesh")
    obj = bpy.data.objects.new(path + "_obj", mesh)
    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh = bpy.context.object.data
    faces = []
    verts = sim["v"]
    edges = sim["c"]
    mesh.from_pydata(verts, edges, faces)
