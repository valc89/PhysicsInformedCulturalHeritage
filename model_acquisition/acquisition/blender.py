"""Blender model acquisition module."""

import os
import bpy
import json

class Blender(object):
    """
    Wrapper to open a Blender ``.blend`` file and extract basic mesh
    information.

    The class opens the given Blender file, collects all objects of type
    ``'MESH'``, and provides utilities to build Python dictionaries containing
    the main object transforms and mesh geometry (vertices, faces, normals).
    """

    def __init__(self, filename : str) -> None:
        """
        Initialize the :class:`Blender` class and open the Blender file.

        :param str filename: Path to the ``.blend`` file to open.
        """
        self.blend_filename = os.path.basename(filename)
        self.blend_name, _ = os.path.splitext(filename)
        bpy.ops.wm.open_mainfile(filepath=filename)  # Apri il file Blender
        self.mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    def _acquire_obj(self, mesh_obj: bpy.types.Object) -> dict:
        """
        Acquire basic information for a single mesh object.

        The returned dictionary contains:
        - object name,
        - object transforms (scale, location, rotation),
        - mesh geometry (triangulated faces, vertices, polygon normals).

        :param bpy.types.Object mesh_obj: The mesh object to acquire.
        :return: A dictionary with the extracted mesh data.
        :rtype: dict
        """
        mesh_data = {}
        mesh_data["name"] = mesh_obj.name  # Aggiungi il nome dell'oggetto

        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='OBJECT')  # Assicurati di essere in modalità oggetto
        
        # Store object scale
        scale = mesh_obj.scale[:]
        mesh_data["scale"] = scale

        # Store object location
        location = mesh_obj.location[:]
        mesh_data["location"] = location

        # Store object rotation
        rotation = mesh_obj.rotation_euler[:]
        mesh_data["rotation"] = rotation
        
        # Convert faces to triangles
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Extract vertices coordinates
        vertices = [v.co[:] for v in mesh_obj.data.vertices]

        # Extract faces (vertex indices) and normals (per polygon)
        faces = []
        normals = []
        for poly in mesh_obj.data.polygons:
            face_vertices = [v for v in poly.vertices]
            # Face normal
            normal = poly.normal
            # If the normal is oriented inward, reverse vertex order
            if normal.dot(mesh_obj.matrix_world.normalized().inverted().transposed().to_3x3() @ poly.center) < 0:
                face_vertices = face_vertices[::-1]
            faces.append(face_vertices)
            normals.append(normal[:])
        
        mesh_data['vertices'] = vertices
        mesh_data['faces'] = faces
        mesh_data['normals'] = normals
        return mesh_data

    
    def acquire_model(self) -> None:
        """
        Acquire mesh data for all mesh objects found in the opened Blender file.

        :return: A list of dictionaries, one for each mesh object.
        :rtype: list[dict]
        """
        mesh_data = [self._acquire_obj(obj) for obj in self.mesh_objects]
        return mesh_data
    
    def export_model(self, mesh_data, filepath) -> None:
        """
        Export acquired mesh data to a JSON file.

        :param list[dict] mesh_data: The mesh data to export.
        :param str filepath: Destination path for the JSON output.
        """
        with open(filepath, 'w') as f:
            json.dump(mesh_data, f)
