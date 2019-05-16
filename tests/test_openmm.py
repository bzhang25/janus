"""
Testing for the openmm_wrapper module
"""
import pytest
from janus.mm_wrapper import OpenMMWrapper
import simtk.unit as OM_unit
import numpy as np
import os

#ala_water_pdb_file = os.path.join(str('tests/files/test_openmm/ala_water.pdb'))
water_pdb_file = os.path.join(str('tests/files/test_openmm/water.pdb'))
ala_pdb_file = os.path.join(str('tests/files/test_openmm/ala_ala_ala.pdb'))

wrapper = OpenMMWrapper(sys_info=water_pdb_file, **{'md_ensemble':'NVT', 'return_info':[]})
wrapper_ala = OpenMMWrapper(sys_info=ala_pdb_file, **{'md_ensemble':'NVT'})

#openmm_mech = openmm_wrapper.OpenMM_wrapper(sys_mech)
#openmm_elec = openmm_wrapper.OpenMM_wrapper(sys_elec)
#openmm_ala_link = openmm_wrapper.OpenMM_wrapper(sys_ala_link)
#openmm_ala_link_2 = openmm_wrapper.OpenMM_wrapper(sys_ala_link)

def test_create_new_residue_template():

    mod = wrapper_ala.create_modeller(keep_atoms=False, atoms=[0,1,2,3])
    wrapper_ala.create_new_residue_template(mod.topology)

    assert wrapper_ala.forcefield._templates['Modified_ALA'].name == 'Modified_ALA' 

def test_set_charge_zero():

    sys1 = wrapper.create_openmm_system(wrapper.pdb.topology)
    sys2 = wrapper.create_openmm_system(wrapper.pdb.topology)

    wrapper.set_charge_zero(sys1)
    wrapper.set_charge_zero(sys2, link_atoms = [0])

    assert sys1.getForce(3).getParticleParameters(0)[0]/OM_unit.elementary_charge == 0.0
    assert sys1.getForce(3).getParticleParameters(1)[0]/OM_unit.elementary_charge == 0.0
    assert sys2.getForce(3).getParticleParameters(0)[0]/OM_unit.elementary_charge == 0.0
    assert sys2.getForce(3).getParticleParameters(1)[0]/OM_unit.elementary_charge == 0.417

def test_set_LJ_zero():

    sys = wrapper.create_openmm_system(wrapper.pdb.topology)

    wrapper.set_LJ_zero(sys)

    assert sys.getForce(3).getParticleParameters(0)[0]/OM_unit.elementary_charge == -0.834
    assert sys.getForce(3).getParticleParameters(0)[1]/OM_unit.nanometer == 0.0
    assert sys.getForce(3).getParticleParameters(0)[2]/OM_unit.kilojoule_per_mole == 0.0

def test_create_openmm_system():
    sys_1 = wrapper.create_openmm_system(wrapper.pdb.topology)
    sys_2 = wrapper.create_openmm_system(wrapper.pdb.topology, include_coulomb='all', initialize=True)
    sys_3 = wrapper.create_openmm_system(wrapper.pdb.topology, include_coulomb='only')

    assert sys_1.getNumForces() == 5
    assert sys_2.getNumForces() == 6
    assert sys_3.getNumForces() == 2

def test_compute_info():
    #print(wrapper.md_ensemble)
    #print(wrapper.integrator)
    state1 = wrapper.compute_info(wrapper.pdb.topology, wrapper.pdb.positions)
    state2 = wrapper.compute_info(wrapper.pdb.topology, wrapper.pdb.positions, minimize=True)

    #print(state1['kinetic'] + state1['potential'])
    #print(state2['kinetic'] + state2['potential'])
    assert np.allclose(state1['kinetic'] + state1['potential'],-0.010557407627282312)
    assert np.allclose(state2['kinetic'] + state2['potential'],-0.02892,rtol=1e-05,atol=1e-05)

def test_initialize():
    wrapper.initialize('Mechanical')
    wrapper_ala.initialize('Electrostatic')
    assert np.allclose(wrapper.main_info['kinetic'] + wrapper.main_info['potential'],  -0.010557407627282312)
    assert np.allclose(wrapper_ala.main_info['kinetic'] + wrapper_ala.main_info['potential'], 0.016526506142315156)

def test_get_main_info():
    state1 = wrapper.get_main_info()
    state2 = wrapper_ala.get_main_info()
    
    assert np.allclose(state1['kinetic'] + state1['potential'], -0.010557407627282312)
    assert np.allclose(state2['kinetic'] + state2['potential'],0.016526506142315156)
    assert 'topology' in state1
    assert 'topology' in state2

def test_take_updated_step():

    force1 = {0 : np.array([0.0,0.0,0.0]), 1 : np.array([0.0, 0.0, 0.0])}
    force2 = {0 : np.array([0.0,0.0,0.0]), 1 : np.array([-0.0001, -0.0001, -0.0001])}
    wrapper.take_updated_step(force1)
    energy1 = wrapper.main_info['kinetic'] + wrapper.main_info['potential']
    #forces1 = wrapper.main_info['forces'][1]

    wrapper.take_updated_step(force2)
    energy2 = wrapper.main_info['kinetic'] + wrapper.main_info['potential']
    #forces2 = wrapper.main_info['forces'][1]

    assert np.allclose(energy1, -0.0105, atol=1e-04)
    assert np.allclose(energy2, -0.009, atol=1e-03)

def test_create_modeller():
    mod1 = wrapper_ala.create_modeller(atoms=[0,1,2,3], keep_atoms=True)
    mod2 = wrapper_ala.create_modeller(atoms=[0,1,2,3], keep_atoms=False)
    atom1 = mod1.topology.getNumAtoms()
    atom2 = mod2.topology.getNumAtoms()

    assert atom1 == 4
    assert atom2 == 29

