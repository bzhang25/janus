from janus.qmmm import AQMMM
from janus.system import System
import numpy as np

class OniomXS(AQMMM):
    """
    Class for the Oniom-XS adaptive QM/MM method.
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

        super().__init__('Oniom-XS', *args, **kwargs)

    def find_configurations(self): 
        """
        Finds the partitions as required by the ONIOM-XS method 
        and saves each partition as a system object.
        Saves all systems in the dictionary self.systems

        Parameters
        ----------
        qm_center : list 
            atoms that define the qm center, default is None

        """
    
        qm = System(qm_indices=self.qm_atoms, qm_residues=self.qm_residues, run_ID=self.run_ID, partition_ID='qm')
        qm.buffer_groups = self.buffer_groups

        self.systems[self.run_ID] = {}
        self.systems[self.run_ID][qm.partition_ID] = qm

        # the following only runs if there are groups in the buffer zone
        if self.buffer_groups:
            qm_bz = System(qm_indices=self.qm_atoms, qm_residues=self.qm_residues, run_ID=self.run_ID, partition_ID='qm_bz')
            for i, buf in self.buffer_groups.items():
                qm_bz.qm_residues.append(i)
                for idx in buf.atoms:
                    qm_bz.qm_atoms.append(idx)

            # each partition has a copy of its buffer groups - 
            qm_bz.buffer_groups = self.buffer_groups

            self.systems[self.run_ID][qm_bz.partition_ID] = qm_bz

    def run_aqmmm(self):
        """
        Interpolates the energy and gradients from each partition
        according to the ONIOM-XS method
        """
        
        qm = self.systems[self.run_ID]['qm']

        if not self.buffer_groups:
            self.systems[self.run_ID]['qmmm_energy'] = qm.qmmm_energy
            self.systems[self.run_ID]['qmmm_forces'] = qm.qmmm_forces

        else:
            qm_bz = self.systems[self.run_ID]['qm_bz']
            lamda = self.get_switching_function()
            

            self.systems[self.run_ID]['qmmm_energy'] = \
            (1- lamda)*qm.qmmm_energy + lamda*qm_bz.qmmm_energy

            # needs work!
            # computing gradients
            print('computing gradients of adaptive')
            forces = {}
            for f, coord in qm_bz.qmmm_forces.items():
                if f in qm.qmmm_forces:
                    forces[f] = lamda*coord + (1-lamda)*qm.qmmm_forces[f] 
                    print('idx, coord, lamda, force')
                    print(f,coord,lamda,forces[f])
                else: 
                    forces[f] = lamda*coord
                    print('idx, coord, lamda, force')
                    print(f,coord,lamda,forces[f])

            # computing gradient of switching function
            scaler = (qm_bz.qmmm_energy - qm.qmmm_energy) / len(qm_bz.buffer_groups)
            print('computing gradients of switching function')

            for i, buf in qm_bz.buffer_groups.items():
                for idx, ratio in buf.weight_ratio.items():
                    forces[idx] += ratio * scaler * buf.d_s_i * buf.COM_coord
                    print('idx, ratio, scaler, force[idx], buf.d_s_i')
                    print(idx,ratio,scaler,forces[idx],buf.d_s_i)
                for idx, ratio in self.qm_center_weight_ratio.items():
                    forces[idx] -= ratio * scaler * buf.d_s_i * buf.COM_coord
                    print('idx, ratio, scaler, force[idx], buf.d_s_i')
                    print(idx,ratio,scaler,forces[idx],buf.d_s_i)

            print('forces after adaptive qm/mm')
            for idx,f in forces.items():
                print(idx,f)
            self.systems[self.run_ID]['qmmm_forces'] = forces


    def get_switching_function(self):
        """
        Averages the individual switching functions 
        of each buffer group

        Returns
        -------
        float
            The average of the switching functions

        """

        s = 0.0
                
        for i, buf in self.buffer_groups.items():
            s += buf.s_i
            
        s *= 1/len(self.buffer_groups)

        return s
            

        
