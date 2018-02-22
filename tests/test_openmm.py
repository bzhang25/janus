"""
Testing for the openmm_wrapper module
"""
import pytest
from janus import openmm_wrapper 
from janus import system
import numpy as np
import os

ala_pdb_file = os.path.join(str('tests/examples/test_openmm/ala_water.pdb'))
water_pdb_file = os.path.join(str('tests/examples/test_openmm/water.pdb'))

qmmm_elec = {"embedding_method" : "Electrostatic"}

sys_ala = system.System(qm={"qm_atoms": [0,1,2,3,4,5,6]}, mm={'mm_pdb_file' : ala_pdb_file})
sys_mech = system.System(qm={"qm_atoms": [0,1,2]}, mm={'mm_pdb_file' : water_pdb_file})
#sys_elec = system.System(qmmm=qmmm_elec, qm=QM, mm=MM)

openmm_ala = openmm_wrapper.OpenMM_wrapper(sys_ala)
openmm_mech = openmm_wrapper.OpenMM_wrapper(sys_mech)
#openmm_elec = openmm_wrapper.OpenMM_wrapper(sys_elec)


#@pytest.mark.datafiles('tests/examples/test_openmm/water.pdb')
#def test_get_state_info(datafiles):
#    """
#    Function to test that get_openmm_energy is getting energy correctly.
#    """
#    sys, pdb = create_system(datafiles, 'water.pdb')
#    sim = janus.openmm_wrapper.create_openmm_simulation(sys, 
#                                                        pdb.topology,
#                                                        pdb.positions)
#    state  = janus.openmm_wrapper.get_state_info(sim)
#    energy = state['potential'] + state['kinetic']
#    assert np.allclose(energy, -0.01056289236)
#
#
#
def test_get_qm_positions():
    """
    Function to test get_qmmm_energy function
    of systems class given the mm energy and the mm energy
    of the qm region
    """
    qm_mol = """O     0.123   3.593   5.841 
 H    -0.022   2.679   5.599 
 H     0.059   3.601   6.796 
 """
    
    pos = openmm_mech.get_qm_positions()

    assert pos == qm_mol

def test_keep_residues():
    """
    Function to test keep_residues function.
    """
    mod_str = openmm_ala.create_modeller()
    mod_int = openmm_ala.create_modeller()
    openmm_wrapper.OpenMM_wrapper.keep_residues(mod_str, ['ALA'])
    openmm_wrapper.OpenMM_wrapper.keep_residues(mod_int, [0,1,2])
    res_str = mod_str.topology.getNumResidues()
    res_int = mod_int.topology.getNumResidues()
    assert res_str == 3 
    assert res_int == 3 

def test_keep_atoms():
    """
    Function to test keep_atoms function.
    """
    qm_atm = openmm_ala._system.qm_atoms 
    mod_str = openmm_ala.create_modeller()
    mod_int = openmm_ala.create_modeller()
    openmm_wrapper.OpenMM_wrapper.keep_atoms(mod_int, qm_atm)
    openmm_wrapper.OpenMM_wrapper.keep_atoms(mod_str, ['N'])
    atom_int = mod_int.topology.getNumAtoms()
    atom_str = mod_str.topology.getNumAtoms()
    assert atom_int == len(qm_atm)
    assert atom_str == 3

def test_delete_residues():
    """
    Function to test delete_residues function.
    """
    mod_str = openmm_ala.create_modeller()
    mod_int = openmm_ala.create_modeller()
    openmm_wrapper.OpenMM_wrapper.delete_residues(mod_str, ['ALA'])
    openmm_wrapper.OpenMM_wrapper.delete_residues(mod_int, [0,1,2])
    res_str = mod_str.topology.getNumResidues()
    res_int = mod_int.topology.getNumResidues()
    assert res_str == 28 
    assert res_int == 28 


def test_delete_atoms():
    """
    Function to test delete_atoms function.
    """
    qm_atm = openmm_ala._system.qm_atoms 
    mod_str = openmm_ala.create_modeller()
    mod_int = openmm_ala.create_modeller()
    openmm_wrapper.OpenMM_wrapper.delete_atoms(mod_int, qm_atm)
    openmm_wrapper.OpenMM_wrapper.delete_atoms(mod_str, ['N'])
    atom_int = mod_int.topology.getNumAtoms()
    atom_str = mod_str.topology.getNumAtoms()
    assert atom_int == 117 - len(qm_atm)
    assert atom_str == 114


def test_delete_water():
    """
    Function to test delete_water function.
    """
    mod = openmm_ala.create_modeller()
    openmm_wrapper.OpenMM_wrapper.delete_water(mod)
    res = mod.topology.getNumResidues()
    atom = mod.topology.getNumAtoms()
    assert res == 3
    assert atom == 33 
