import mcubes
import trimesh
import pyrender
import cv2
import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
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
        vertices *= 0.5
        vertices -= (grid_dims)*0.5 + np.array([0,0,grid_dims[-1]])
        if len(vertices):
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            color = np.array([0,0,0,alpha])
            color[:-1] = ATOM_COLORS[atom]
            mesh.visual.vertex_colors = color#ATOM_COLORS[atom]
            meshes.append(pyrender.Mesh.from_trimesh(mesh))
    return meshes


def transform_coords(coords):
    grid_dims = [TMCfg.grid_dim]*3
    trans = coords - np.array([0, 0, grid_dims[-1]])
    return np.array([trans[1], trans[0], trans[2]])

def make_bond_cyl(coord1, coord2, bond_type, meshes, alpha):
    coord1 = coord1.detach().cpu().numpy()
    coord2 = coord2.detach().cpu().numpy()
    diff = coord1 - coord2
    if np.all(diff == 0): return

    diff = diff/np.linalg.norm(diff)
    if diff[0] == 0:
        offset = np.cross(diff, [1, 0, 0])
    else:
        offset = np.cross(diff, [0, 1, 0])
    offset = offset/np.linalg.norm(offset)
    if bond_type == BOND_TYPE_HASH[Chem.BondType.SINGLE]:
        dists = [0.0]
    elif bond_type == BOND_TYPE_HASH[Chem.BondType.AROMATIC]:
        dists = np.linspace(-0.2, 0.2, 2)
    else:
        dists = np.linspace(-0.2, 0.2, bond_type)
    for k in range(len(dists)):
        start = transform_coords(coord1 + offset*dists[k])
        end = transform_coords(coord2 + offset*dists[k])
        if bond_type == BOND_TYPE_HASH[Chem.BondType.AROMATIC] and k == 0:
            coef = 0.45
            mid2 = start*coef + end*(1-coef)
            mid1 = start*(1-coef) + end*coef
            for s, e in [(start, mid1), (mid2, end)]:
                sm = trimesh.creation.cylinder(radius=0.1, segment=[s, e])
                color = np.array([127,127,127,alpha])
                sm.visual.vertex_colors = color
                meshes.append(pyrender.Mesh.from_trimesh(sm))
        else:
            sm = trimesh.creation.cylinder(radius=0.1, segment=[start, end])
            color = np.array([127,127,127,alpha])
            sm.visual.vertex_colors = color
            meshes.append(pyrender.Mesh.from_trimesh(sm))

def get_kp_meshes(tmol, alpha=255):

    coords = tmol.get_coords()
    meshes = []
    for i, enc in enumerate(tmol.atom_types):
        if enc in [ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^']]:
            continue
        sm = trimesh.creation.uv_sphere(radius=ATOM_RADII_LIST[enc]/4)
        color = np.array([0,0,0,alpha])
        color[:-1] = ATOM_COLORS[ATOM_TYPE_LIST[enc]]
        sm.visual.vertex_colors = color
        tfs = np.tile(np.eye(4), (1, 1, 1))
        tfs[:,:3,3] = transform_coords(coords[i])
        meshes.append(pyrender.Mesh.from_trimesh(sm, poses=tfs))
        if tmol.bonds is None and i > 0:
            make_bond_cyl(coords[i], coords[i-1], BOND_TYPE_HASH[Chem.BondType.SINGLE], meshes, alpha)
    if tmol.bonds is not None:
        for start, end, bond in tmol.bonds.get_all_indexes():
            if tmol.atom_types[start] not in [ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^']] and tmol.atom_types[end] not in [ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^']]:
                make_bond_cyl(coords[start], coords[end], bond, meshes, alpha)
        
        
    return meshes

def get_molgrid_scene(tmol, prot_mg=None, thresh=0.5):
    meshes = get_molgrid_meshes(tmol, 255, thresh)
    if prot_mg is not None:
        meshes += get_molgrid_meshes(prot_mg, 128, thresh)
    scene = pyrender.Scene()
    for mesh in meshes:
        scene.add(mesh)
    return scene

def get_kp_scene(tmol):
    meshes = get_kp_meshes(tmol)
    #meshes += get_molgrid_meshes(tmol, 128, 0.5)
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

def render_kp(tmol, dims=(300,300)):
    scene = get_kp_scene(tmol)
    return scene2img(scene, dims)

def render_kp_rt(tmol, prot_mg=None):
    scene = get_kp_scene(tmol)
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
    if tmol.atom_types is None:
        return np.full((*dims, 3), 255, dtype=np.uint8)
    tmola = tmol.argmax()
    imgs = []
    if tmol_template.molgrid is not None:
        imgs.append(render_molgrid(tmola, dims=dims))
    if tmol_template.atom_types is not None:
        if tmol_template.kps is not None or tmol_template.kps_1h is not None:
            imgs.append(render_kp(tmola, dims))
        else:
            imgs.append(render_text(tmola.atom_str(), dims))
    if tmol.bonds is not None:
        imgs.append(cv2.cvtColor(np.array(Draw.MolToImage(tmola.get_mol(False), kekulize=False, size=dims)), cv2.COLOR_BGR2RGB))
    return get_multi([[img] for img in imgs])

def test_molgrid():
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    render_molgrid_rt(tm)

def test_kp():
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    tm = tm.argmax()
    render_kp_rt(tm)

def test_render_tmol():
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    rdMolTransforms.TransformConformer(mol.GetConformer(0), rand_rotation_matrix())
    tm = TensorMol(mol)
    img = render_tmol(tm)
    cv2.imshow("mol", img)
    cv2.waitKey(0)

def test_uff():
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    tm1 = TensorMol(mol)
    sz = TMCfg.grid_size
    # batch, atom_idx, width, height, depth
    kp_shape = (TMCfg.max_atoms, sz, sz, sz)
    #tm1.kps_1h = torch.zeros(kp_shape)
    mol = tm1.get_mol()
    Chem.SanitizeMol(mol)
    #mol = Chem.AddHs(mol)
    #print(AllChem.EmbedMolecule(mol))
    tm1 = TensorMol(mol)
    for i in range(mol.GetNumAtoms()):
        print(i, mol.GetConformer().GetAtomPosition(i).x,
              mol.GetConformer().GetAtomPosition(i).y,
              mol.GetConformer().GetAtomPosition(i).z)
    print(AllChem.UFFOptimizeMolecule(mol))
    for i in range(mol.GetNumAtoms()):
        print(i, mol.GetConformer().GetAtomPosition(i).x,
              mol.GetConformer().GetAtomPosition(i).y,
              mol.GetConformer().GetAtomPosition(i).z)
    tm2 = TensorMol(mol)
    print((tm1.get_coords() == tm2.get_coords()).all())

    mol_orig = tm1.get_mol()
    print(Chem.rdMolAlign.AlignMol(mol, mol_orig))
    
    meshes = get_kp_meshes(tm1)
    meshes += get_kp_meshes(tm2, 128)
    scene = pyrender.Scene()
    for mesh in meshes:
        scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tensor_mol import TensorMol
    from utils import rand_rotation_matrix
    from rdkit.Chem import rdMolTransforms, AllChem, rdMolAlign
    cfg = OmegaConf.create({
        'grid_dim': 16,
        'grid_step': 0.5,
        'max_atoms': 38,
        'max_valence': 6
    })
    TMCfg.set_cfg(cfg)
    #test_uff()
    #test_render_tmol()
    test_kp()
