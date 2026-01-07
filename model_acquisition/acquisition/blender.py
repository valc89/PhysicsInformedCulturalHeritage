import os
import bpy
import json

import os
import bpy
import json

class Blender(object):
    """
        Classe per l'acquisizione dei modelli 3D
    """

    def __init__(self, filename : str):
        self.blend_filename = os.path.basename(filename)
        self.blend_name, _ = os.path.splitext(filename)
        bpy.ops.wm.open_mainfile(filepath=filename)  # Apri il file Blender
        self.mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    def _acquire_obj(self, mesh_obj: bpy.types.Object) -> dict:
        mesh_data = {}
        mesh_data["name"] = mesh_obj.name  # Aggiungi il nome dell'oggetto

        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='OBJECT')  # Assicurati di essere in modalità oggetto
        
        # Memorizza le scale dell'oggetto
        scale = mesh_obj.scale[:]
        mesh_data["scale"] = scale

        # Memorizza la posizione dell'oggetto
        location = mesh_obj.location[:]
        mesh_data["location"] = location

        # Memorizza la rotazione dell'oggetto
        rotation = mesh_obj.rotation_euler[:]
        mesh_data["rotation"] = rotation
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        vertices = [v.co[:] for v in mesh_obj.data.vertices]
        faces = []
        normals = []
        for poly in mesh_obj.data.polygons:
            face_vertices = [v for v in poly.vertices]
            # Calcola la normale della faccia
            normal = poly.normal
            # Verifica l'orientamento della normale
            if normal.dot(mesh_obj.matrix_world.normalized().inverted().transposed().to_3x3() @ poly.center) < 0:
                face_vertices = face_vertices[::-1]  # Inverti l'ordine dei vertici se la normale è rivolta verso l'interno
            faces.append(face_vertices)
            normals.append(normal[:])
        
        mesh_data['vertices'] = vertices
        mesh_data['faces'] = faces
        mesh_data['normals'] = normals
        return mesh_data

    
    def acquire_model(self):
        mesh_data = [self._acquire_obj(obj) for obj in self.mesh_objects]
        return mesh_data
    
    def export_model(self, mesh_data, filepath):
        with open(filepath, 'w') as f:
            json.dump(mesh_data, f)
