from pymol import cmd, preset

MOL_TMP_FILE = './test_output/mol_tmp.sdf'
PNG_TMP_FILE = './test_output/mol_tmp.png'

dims = (300, 300)
cmd.load(MOL_TMP_FILE)
cmd.zoom()
preset.ball_and_stick(selection='all', mode=1)
cmd.bg_color('black')
cmd.set('ray_opaque_background', 1)
cmd.png(PNG_TMP_FILE, *dims)
