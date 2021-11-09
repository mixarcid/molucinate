import hydra
import cv2
import trimesh
import pyrender
from tqdm import tqdm

from data.chem import *
from data.make_dataset import make_dataset
from data.tensor_mol import TMCfg, TensorMol
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from data.render import *

@hydra.main(config_path='cfg', config_name="config")
def represent(cfg):

    TMCfg.set_cfg(cfg.data)

    test_d = make_dataset(cfg, False)
    (tmol, _), __ = next(iter(test_d))
    
    img = render_tmol(tmol, dims=(600, 600))
    cv2.imwrite("slides/mol.png", img)
    return

    coords = tmol.get_coords()
    alpha = 255
    meshes = []
    for i, enc in enumerate(tmol.atom_types):
        if enc in [ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^']]:
            break
        sm = trimesh.creation.uv_sphere(radius=ATOM_RADII_LIST[enc]/4)
        color = np.array([0,0,0,alpha])
        color[:-1] = ATOM_COLORS[ATOM_TYPE_LIST[enc]]
        sm.visual.vertex_colors = color
        tfs = np.tile(np.eye(4), (1, 1, 1))
        tfs[:,:3,3] = transform_coords(coords[i])
        meshes.append(pyrender.Mesh.from_trimesh(sm, poses=tfs))

    for i, mesh in tqdm(enumerate(meshes)):
        scene = pyrender.Scene()
        scene.add(mesh)
        atom_img = scene2img(scene, (600, 600))
        cv2.imwrite(f"slides/mol_{i}.png", atom_img)

if __name__ == "__main__":
    represent()
