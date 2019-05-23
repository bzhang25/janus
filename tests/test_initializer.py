import pytest
from janus import initializer
from janus.mm_wrapper import OpenMMWrapper
import mdtraj as md
import numpy as np
import os


water = os.path.join(str('tests/files/test_openmm/water.pdb'))
param1 = {"system" : {"system_info" : [water],
                      "aqmmm_scheme" : "ONIOM-XS",
                      "run_aqmmm" : True},

          "qmmm" : {"qm_atoms" : [0,1,2]},
          "aqmmm" : {"qm_center" : [0]}
         }
param2 = {
            "system" : {"system_info" : water,
                      "aqmmm_scheme" : "Hot-Spot",
                      "run_aqmmm" : True,},

            "aqmmm" : {
                        "partition_scheme" : "distance",
                        "qm_center" : [0]},

            "qmmm" : {"embedding_method" : "Electrostatic",
                      "qm_atoms" : [0,1,2]}}

param3 = os.path.join(str('tests/files/test_initializer/input.json'))

init1 = initializer.Initializer(param1, as_file=False)
init2 = initializer.Initializer(param2, as_file=False)
init3 = initializer.Initializer(param3)

def test_initialize_wrappers():

    mm1, qmmm1, t1 = init1.initialize_wrappers()
    mm2, qmmm2, t2 = init2.initialize_wrappers()
    mm3, qmmm3, t3 = init3.initialize_wrappers()

    assert qmmm1.class_type == 'Oniom-XS'
    assert qmmm2.class_type == 'Hot-Spot'
    assert mm1.class_type == 'OpenMM' 
    assert mm2.class_type == 'OpenMM' 
    assert init3.md_sim_wrapper is OpenMMWrapper
    assert qmmm3.class_type == 'QMMM'
    assert mm3.class_type == 'OpenMM' 
