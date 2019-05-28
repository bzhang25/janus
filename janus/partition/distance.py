import numpy as np
from copy import deepcopy
import mdtraj as md
from janus.partition import Partition

class DistancePartition(Partition):

    def __init__(self, trajectory, topology, Rmin, Rmax):

        self.Rmin = Rmin
        self.Rmax = Rmax

        super().__init__(trajectory, topology, 'distance')

    def define_buffer_zone(self, qm_center, qm_center_residues, prev_qm=None, prev_bf=None):
        """
        Determines buffer group atoms.
        Gets the buffer groups in the buffer zone based on a distance 
        partitioning scheme, saves each buffer group as a 
        :class:`~janus.system.Buffer` object,
        and saves all buffer groups in the dictionary self.buffer_groups.
        For water as a solvent, considers the whole water molecule as a buffer group.

        Note
        ----
        Currently only worked with explicit solvent based systems. 
        Cannot treat buffer atoms that are part of large molecular structures (e.g., proteins).

        Parameters
        ----------
        qm_center : list 
            the indicies that make up the qm center

        """

        self.qm_residues = []
        self.buffer_groups = {}
        residue_tracker = [] 
        top = self.topology

        
        self.find_buffer_atoms(qm_center)

        for i in self.buffer_atoms:
            idx = top.atom(i).residue.index

            if (idx not in residue_tracker and idx not in qm_center_residues):
                residue_tracker.append(idx)
                buf = self.get_residue_info(idx)
            
                if buf.r_i < self.Rmin:
                    self.edit_atoms(atoms=self.qm_atoms, res_idx=idx, add=True)
                    self.qm_residues.append(idx)
                    
                elif buf.r_i >= self.Rmax:
                    self.edit_atoms(atoms=self.qm_atoms, res_idx=idx, remove=True)

                elif (buf.r_i >= self.Rmin and buf.r_i < self.Rmax):
                    self.buffer_groups[idx] = buf
                    self.edit_atoms(atoms=self.qm_atoms, res_idx=idx, remove=True)

        #print('qm_atoms after buffer editing: {}'.format(self.qm_atoms))
        qm_atoms = deepcopy(self.qm_atoms)
        # tracking qm_residues and cleaning up qm
        for i in qm_atoms:
            idx = top.atom(i).residue.index
            if idx not in self.qm_residues:
                if idx in qm_center_residues:
                    self.edit_atoms(atoms=self.qm_atoms, res_idx=idx, add=True)
                    self.qm_residues.append(idx)
                else:
                    res = self.get_residue_info(idx)
                    if res.r_i >= self.Rmax:
                        self.edit_atoms(atoms=self.qm_atoms, res_idx=idx, remove=True)
                    elif res.r_i < self.Rmin:
                        self.edit_atoms(atoms=self.qm_atoms, res_idx=idx, add=True)
                        self.qm_residues.append(idx)
                    elif (res.r_i >= self.Rmin and res.r_i < self.Rmax):
                        self.buffer_groups[idx] = res
                        self.edit_atoms(atoms=self.qm_atoms, res_idx=idx, remove=True)

        #print('qm_atoms after final editing: {}'.format(self.qm_atoms))
        # fragment
        #print('checking for fragments')
        #self.find_fragments()
        

    def find_buffer_atoms(self, qm_center):
        """
        Find the buffer groups whose COM falls in between Rmin and Rmax

        Parameters
        ----------
        qm_center : list 
            the indicies that make up the qm center

        """
        temp_traj, qm_center_idx = self.compute_qm_center_info(qm_center)

        rmin_atoms = md.compute_neighbors(temp_traj, self.Rmin/10, qm_center_idx)
        rmax_atoms = md.compute_neighbors(temp_traj, self.Rmax/10, qm_center_idx)
        self.buffer_atoms = np.setdiff1d(rmax_atoms, rmin_atoms)
        self.qm_atoms = rmin_atoms[0].tolist()
        print('qm atoms found by mdtraj: {}'.format(self.qm_atoms))
        print('buffer atoms found by mdtraj: {}'.format(self.buffer_atoms))

        if self.COM_as_qm_center is False:
            self.qm_atoms.append(qm_center[0])


    def get_Rmin(self):
        """
        Function to return self.Rmin

        Returns
        -------
        float
            the distance from qm center to inner limit of buffer zone in angstroms

        """
        return self.Rmin

    def get_Rmax(self):
        """
        Function to return self.Rmin

        Returns
        -------
        float
            the distance from qm center to outer limit of buffer zone in angstroms

        """

        return self.Rmax

    def set_Rmin(self, Rmin):
        """
        Function to set self.Rmin
        
        Parameters
        ----------
        Rmin : float 
            the distance from qm center to inner limit of buffer zone in angstroms
        
        """

        self.Rmin = Rmin

    def set_Rmax(self, Rmax):
        """
        Function to set self.Rmax
        
        Parameters
        ----------
        Rmax : float 
            the distance from qm center to outer limit of buffer zone in angstroms
        
        """
        self.Rmax = Rmax

