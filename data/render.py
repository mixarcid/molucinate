import mcubes
import trimesh
import pyrender
import cv2
import os
import numpy as np
import torch
try:
    from .chem import *
    from .tensor_mol import TMCfg
except ImportError:
    from chem import *
    from tensor_mol import TMCfg

should_egl = True
try:
    if os.environ["DISPLAY"] == ":0":
        should_egl = False
except KeyError:
    pass
if should_egl:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

def get_molgrid_meshes(tmol, alpha, thresh):
    molgrid = tmol.molgrid.cpu().numpy()
    grid_dims = np.array([TMCfg.grid_dim, TMCfg.grid_dim, TMCfg.grid_dim])
    meshes = []
    for atom in ATOM_TYPE_LIST:
        if atom in '_^': continue
        atoms = molgrid[ATOM_TYPE_HASH[atom]] > thresh
        smoothed = atoms#mcubes.smooth(atoms)
        vertices, triangles = mcubes.marching_cubes(smoothed, 0)
        vertices -= grid_dims + np.array([0,0,grid_dims[-1]])
        if len(vertices):
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            color = np.array([0,0,0,alpha])
            color[:-1] = ATOM_COLORS[atom]
            mesh.visual.vertex_colors = color#ATOM_COLORS[atom]
            meshes.append(pyrender.Mesh.from_trimesh(mesh))
    return meshes

def get_molgrid_scene(tmol, prot_mg=None, thresh=0.5):
    meshes = get_molgrid_meshes(tmol, 255, thresh)
    if prot_mg is not None:
        meshes += get_molgrid_meshes(prot_mg, 128)
    scene = pyrender.Scene()
    for mesh in meshes:
        scene.add(mesh)
    return scene

def scene2img(scene, dims):
    pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    scene.add(pc)
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(dl)
    r = pyrender.OffscreenRenderer(viewport_width=dims[0],
                                   viewport_height=dims[1],
                                   point_size=1.0)
    color, depth = r.render(scene)
    r,g,b = cv2.split(color)
    img = cv2.merge((b,g,r))
    return img

def render_molgrid(tmol, prot_mg=None, thresh=0.5, dims=(300,300)):
    scene = get_molgrid_scene(tmol, prot_mg, thresh)
    return scene2img(scene, dims)

def render_molgrid_rt(tmol, prot_mg=None):
    scene = get_molgrid_scene(tmol, prot_mg)
    pyrender.Viewer(scene, use_raymond_lighting=True)

def get_multi(arr):
    num_y = len(arr)
    num_x = len(arr[0])
    base_y, base_x, channels = arr[0][0].shape
    out_shape = (num_y*base_y, num_x*base_x, channels)
    out = np.zeros(out_shape, dtype=np.uint8)
    for i, row in enumerate(arr):
        for j, img in enumerate(row):
            out[i*base_y:(i+1)*base_y, j*base_x:(j+1)*base_x] = img
    return out

def export_multi(fname, arr):
    img = get_multi(arr)
    cv2.imwrite(fname, img)

def render_text(text, dims):
    scale = 0.5
    img = np.full((*dims, 3), 255, dtype=np.uint8)
    line_width = 12
    num_lines = len(text)//line_width + 1
    offset = 15
    line_height = 15
    for i in range(num_lines):
        end = min((i+1)*line_width, len(text))
        line = text[(i*line_width):end]
        ydim = dims[1]-line_height*(num_lines-i)-offset
        img = cv2.putText(img, line,
                          (offset, ydim),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          scale, (0,0,0), 1, cv2.LINE_AA)
    return img

def render_tmol(tmol, tmol_template=None, dims=(300,300)):
    if tmol_template is None:
        tmol_template = tmol
    tmola = tmol.argmax()
    imgs = []
    if tmol_template.molgrid is not None:
        imgs.append(render_molgrid(tmol))
    if tmol_template.atom_types is not None:
        imgs.append(render_text(tmola.atom_str(), dims))
    return get_multi([imgs])

def test_molgrid():
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    render_molgrid_rt(tm)

def test_render_tmol():
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    tm = TensorMol(mol)
    img = render_tmol(tm)
    cv2.imshow("mol", img)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tensor_mol import TensorMol
    cfg = OmegaConf.create({
        'grid_dim': 16,
        'grid_step': 0.5,
        'max_atoms': 38,
        'max_valence': 6
    })
    TMCfg.set_cfg(cfg)
    test_render_tmol()
