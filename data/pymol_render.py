import pymol2

MOL_TMP_FILE = './test_output/mol_tmp.sdf'
PNG_TMP_FILE = './test_output/mol_tmp.png'

with pymol2.PyMOL() as pm:
    pm.cmd.load(MOL_TMP_FILE)
    pm.cmd.zoom()
    pm.preset.ball_and_stick(selection='all', mode=1)
    pm.cmd.bg_color('black')
    pm.cmd.set('ray_opaque_background', 1)
    pm.cmd.png(PNG_TMP_FILE, *dims)
