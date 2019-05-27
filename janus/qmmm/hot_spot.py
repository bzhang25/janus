from janus.qmmm import AQMMM
from janus.system import System
import numpy as np
from copy import deepcopy

class HotSpot(AQMMM):
    """
    Class for the Hot Spot adaptive QM/MM method.
    Inherits from AQMMM class

    Parameters
    ----------
        hl_wrapper : :class:`~janus.mm_wrapper.MMWrapper` subclass or :class:`~janus.qm_wrapper.QMWrapper` subclass
            Wrapper for performing the high-level computation. 
            Traditionally QM but user can define MM.
        ll_wrapper : :class:`~janus.mm_wrapper.MMWrapper` subclass
            Wrapper for performing the low-level computation
        sys_info : str 
            A string with the filename or a list with multiple filenames 
            that contain position and topology information. 
        sys_info_format : str 
            Describes what kind of input is contained in sys_info. Default is pdb.
        qm_center: list 
            Atoms that define the qm center, default is [0].
            If more than one index is given, COM is used as qm_center
        partition_scheme: str 
            Scheme to use to define buffer groups,
            default is distance (only scheme available as of now)
        Rmin: float 
            Inner radius for distance partition in angstroms, default is 3.8
        Rmax: float 
            Outer radius for distance partition in angstroms, default is 4.5
        qmmm_param : dict
            A dictionary with any parameters to pass into the QMMM class.
            See QMMM class for specifics

    """
    def __init__(self, *args, **kwargs):

        super().__init__('Hot-Spot', *args, **kwargs)

    def find_configurations(self): 
        """
        Finds the partitions as required by the Hot-Spot method 
        and saves each partition as a system object.
        Saves all systems in the dictionary self.systems

        Parameters
        ----------
        qm_center : list 
            atoms that define the qm center, default is None

        """
    
        qm = System(qm_indices=self.qm_atoms, qm_residues=self.qm_residues, run_ID=self.run_ID, partition_ID='qm')
        qm.buffer_groups = self.buffer_groups

        # the following only runs if there are groups in the buffer zone
        if self.buffer_groups:

            qm.buffer_groups = deepcopy(self.buffer_groups)

            for i, buf in self.buffer_groups.items():
                qm.qm_residues.append(i)
                for idx in buf.atoms:
                    qm.qm_atoms.append(idx)
                
        self.systems[self.run_ID] = {}
        self.systems[self.run_ID][qm.partition_ID] = qm

    def run_aqmmm(self):
        """
        Interpolates the energy and gradients from each partition
        according to the Hot-Spot method
        """
        
        qm = self.systems[self.run_ID]['qm']
        self.systems[self.run_ID]['qmmm_energy'] = qm.qmmm_energy

        entire_grad = qm.entire_sys['gradients']
        ps_mm_grad, qm_grad = qm.primary_subsys['ll']['gradients'], qm.primary_subsys['hl']['gradients']
        forces = {}

        buf_atoms = {}
        if self.buffer_groups:
            for i, buf in self.buffer_groups.items():
                for idx in buf.atoms:
                    buf_atoms[idx] = buf.s_i
                
        for i, atom in enumerate(self.qm_atoms):
            if atom in buf_atoms.keys():
                #print('forces from buf atom {} with s_i {}'.format(atom,buf_atoms[atom]))
                forces[atom] = -(-1*entire_grad[atom]) + buf_atoms[atom]*(-1*qm_grad[i]) + (1-buf_atoms[atom])*(-1*ps_mm_grad[i])

            else:
                #print('forces from qm atom')
                forces[atom] = -(-1*entire_grad[atom]) + (-1*qm_grad[i])

            self.systems[self.run_ID]['qmmm_forces'] = forces

    def compute_lamda_i(self, r_i):
        """
        Computes the switching function of the Hot-Spot method
        and overrides the compute_lamda_i function from the AQMMM class
        
        Parameters
        ----------
        r_i : float 
            the distance between the qm center and the COM in angstroms 

        Returns
        -------
        float
            lamda_i, unitless
        None
            placeholder for the derivative of r_i
        
        """

        if r_i <= self.Rmin:
            lamda_i = 1


        elif r_i > self.Rmax:
            lamda_i = 0

        else:
            lamda_i = (self.Rmax**2 - r_i**2)**2 * (self.Rmax**2 + 2*(r_i**2) - 3*(self.Rmin**2))
            lamda_i *= 1/(self.Rmax**2 - self.Rmin**2)**3

        return lamda_i, None
