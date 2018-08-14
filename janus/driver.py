"""
This is the qmmm driver module
"""
import json
from . import parser
from .system import System
from .qm_wrapper import QM_wrapper 
from .psi4_wrapper import Psi4_wrapper 
from .mm_wrapper import MM_wrapper 
from .openmm_wrapper import OpenMM_wrapper 
from .qmmm import QMMM
from .aqmmm import AQMMM
from .oniom_xs import ONIOM_XS


def load_system(filename):

    with open(filename) as parameter_file:
        parameters = json.load(parameter_file)

    system = System(parameters['aqmmm'], parameters['qmmm'], parameters['qm'], parameters['mm'])
    
    return system

def initialize_wrappers(system, aqmmm=False):
    """
    Initializes the programs to use for computations
    """

    if aqmmm is False:
        # create qm_wrapper object
        if system.qm_program == "Psi4":
            qm_wrapper = Psi4_wrapper(system)
        else:
        # add other options for qm program here
            print("Only Psi4 currently available")

        # create mm_wrapper object
        if system.mm_program == "OpenMM":
            mm_wrapper = OpenMM_wrapper(system)
        else:
        # add other options for mm program here
            print("Only OpenMM currently available")

        return mm_wrapper, qm_wrapper

    if aqmmm is True:
        if system.aqmmm_scheme == 'ONIOM-XS':
            aqmmm = ONIOM_XS(system.aqmmm, system.mm_pdb_file)
        else:
            print("Only ONIOM_XS currently implemented")
        
        return aqmmm


def run_adaptive(system):

    # initialize wrappers
    mm_wrapper, qm_wrapper = initialize_wrappers(system)

    aqmmm = initialize_wrappers(system, aqmmm=True)

    qmmm = QMMM(qm_wrapper)

    # initialize mm_wrapper with information about initial system
    mm_wrapper.initialize_system()

    for step in range(system.steps):

        #get MM information for entire system
        main_info = mm_wrapper.get_main_info()

        # main info will have positions and topology to update trajectory
        partitions = aqmmm.partition(info=main_info)

        for i, partition in partitions.items():
            print(partition.qm_atoms)
            qmmm.get_info(system.qmmm_scheme, mm_wrapper, partition=partition)
            aqmmm.save(partition.ID, qmmm.qmmm_forces, qmmm.qmmm_energy)
            
        # get aqmmm forces 
        forces = aqmmm.get_info()
    
        # feed forces into md simulation and take a step
        # make sure positions are updated so that when i get information on entire system 
        # getting it on the correct one
        mm_wrapper.take_step(force=forces)

def run_qmmm(system):
# have this as part of run_adaptive?

    # initialize wrappers
    mm_wrapper, qm_wrapper = initialize_wrappers(system)

    qmmm = QMMM(qm_wrapper)

    # initialize mm_wrapper with information about initial system
    mm_wrapper.initialize_system()

    for step in range(system.steps):

        qmmm.get_info(system.qmmm_scheme, mm_wrapper)
            
        # get qmmm forces 
        forces = qmmm.qmmm_forces 
    
        # feed forces into md simulation and take a step
        # make sure positions are updated so that when i get information on entire system 
        # getting it on the correct one
        mm_wrapper.take_step(force=forces)

