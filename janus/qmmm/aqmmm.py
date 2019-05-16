from abc import ABC, abstractmethod
from copy import deepcopy
import mdtraj as md
import numpy as np
from janus.partition import DistancePartition, HystereticPartition
from janus.qmmm import QMMM

class AQMMM(ABC, QMMM):
    """
    AQMMM super class for adaptive QMMM computations.
    Inherits from QMMM class.

    Note
    ----
    Since AQMMM is a super class and has abstract methods
    cannot actually instantiate AQMMM object, but only its child objects
    """


    def __init__(self, class_type,
                       hl_wrapper, 
                       ll_wrapper, 
                       sys_info,
                       sys_info_format='pdb',
                       qm_center=[0],
                       partition_scheme='distance',
                       Rmin=3.8,
                       Rmax=4.5,
                       Rmin_qm=3.6,
                       Rmin_bf=4.3,
                       qmmm_param={}):


        super().__init__(hl_wrapper, ll_wrapper, sys_info, sys_info_format=sys_info_format, **qmmm_param)

        self.qm_center = qm_center
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Rmin_qm = Rmin_qm
        self.Rmin_bf = Rmin_bf
        self.class_type = class_type
        self.buffer_groups = {}
        self.get_qm_center_residues()
        self.compute_zero_energy()

        self.buffer_wrapper =  self.get_buffer_wrapper(partition_scheme)

    def run_qmmm(self, main_info, wrapper_type):
        """
        Drives QM/MM computation.
        Updates the positions and topology given in main_info,
        determines the partitions to be computed, for each partition,
        determines the QM/MM energy and gradients, then interpolates all
        partitions using specified adaptive QM/MM scheme

        Parameters
        ----------
        main_info : dict 
            Contains the energy, forces, topology, and position information 
            for the whole system
        wrapper_type : str
            Defines the program used to obtain main_info
        """

        self.update_traj(main_info['positions'], main_info['topology'], wrapper_type)
        self.buffer_wrapper.update_traj(self.traj)
        self.find_buffer_zone()
        self.find_configurations()

        counter = 0
        for i, system in self.systems[self.run_ID].items():
            print('Running QM/MM partition {}'.format(counter))
            print('Number of QM atoms for partition {} is {}'.format(counter,len(system.qm_atoms)))

            self.qm_atoms = deepcopy(system.qm_atoms)

            if self.embedding_method =='Mechanical':
                self.mechanical(system, main_info)
            elif self.embedding_method =='Electrostatic':
                self.electrostatic(system, main_info)
            else:
                print('only mechanical and electrostatic embedding schemes implemented at this time')
            counter += 1

        print('QM/MM partitions done. Getting zero energies')
        self.get_zero_energy()
        print('Interpolating QM/MM partitions')
        self.run_aqmmm()
        self.systems[self.run_ID]['kinetic_energy'] = main_info['kinetic']
        #print('!qmmm_energy', self.systems[self.run_ID]['qmmm_energy'])
        #if self.run_ID % 10 == 0:
        print('!', self.run_ID, self.systems[self.run_ID]['qmmm_energy'] + self.systems[self.run_ID]['kinetic_energy'])

        # updates current step count
        self.run_ID += 1

        # delete the information of 2 runs before, only save current run and previous run information at a time
        if self.run_ID > 1:
            del self.systems[self.run_ID - 2]

        
    def compute_lamda_i(self, r_i):
        """
        Computes the switching function and the derivative 
        of the switching function defined as a 5th order spline:
        
        .. math::
            \lambda_i = -6x^5 + 15x^4 - 10x^3 + 1 
            d_{\lambda_i} = -30x^4 + 60x^3 - 30x^2

        where x is the reduced distance (r_i - rmin)/(rmax - rmin)
        of buffer group i

        Parameters
        ----------
        r_i : float 
            the distance between the qm center and the COM (in angstroms)

        Returns
        -------
        float
            lamda_i, unitless
        float
            derivative of lamda_i, unitless

        """

        x_i = float((r_i - self.Rmin) / (self.Rmax - self.Rmin))

        if (x_i < 0 or x_i > 1):
            raise ValueError("reduced distance x_i has to be between 0 and 1")

        lamda_i = -6*((x_i)**5) + 15*((x_i)**4) - 10*((x_i)**3) + 1

        d_lamda_i = (-30*(x_i)**4  + 60*(x_i)**3 - 30*(x_i)**2)
        d_lamda_i *= 1/(r_i * (self.Rmax - self.Rmin))

        return lamda_i, d_lamda_i


    def compute_zero_energy(self):
        """
        Compute the energy of the isolated groups at their minimum geometry
        """
 
        # for explicit solvent systems can just do once, but for bond forming/breaking processes
        # need to update??
        # this is only functional for explicitly solvated systems
        
        # get all the unique groups 
        
        residues = {'qm_center' : []}
        for res in self.topology.residues:
            if res.index in self.qm_center_residues:
                for atom in res.atoms:
                    residues['qm_center'].append(atom.index)
            else:
                if res.name not in residues:
                    residues[res.name] = []
                    for atom in res.atoms:
                        residues[res.name].append(atom.index)

        self.qm_zero_energies = {}
        self.mm_zero_energies = {}

        for res in residues:
            traj = self.traj.atom_slice((residues[res]))

            mm = self.ll_wrapper.get_energy_and_gradient(traj, minimize=True)
            self.mm_zero_energies[res] = mm['energy']

            qm = self.hl_wrapper.get_energy_and_gradient(traj, minimize=True)
            self.qm_zero_energies[res] = qm['energy']

    def get_qm_center_residues(self):
        residues = []
        for atom in self.qm_center:
            res = self.topology.atom(atom).residue
            if res not in residues:
                residues.append(res.index)
        self.qm_center_residues = residues
    
    def get_zero_energy(self):
        """
        Incorporates the zero energy of groups to the total qmmm energy
        """

        print('step', self.run_ID)
        for i, sys in self.systems[self.run_ID].items():
            print(sys.qm_residues)
            print(sys.qm_atoms)
            print(sys.qmmm_energy)
            sys.zero_energy += self.qm_zero_energies['qm_center']
            for res in self.topology.residues:
                if (res.index in sys.qm_residues and res.index not in self.qm_center_residues):
                    sys.zero_energy += self.qm_zero_energies[res.name]
                elif (res.index not in sys.qm_residues and res.index not in self.qm_center_residues):
                    sys.zero_energy += self.mm_zero_energies[res.name]
            print(sys.zero_energy)

            # maybe I should save a separate copy of qmmm energy somewhere
            sys.qmmm_energy -= sys.zero_energy

    def get_buffer_wrapper(self, partition_scheme):

        if partition_scheme == 'distance':
            wrapper = DistancePartition(self.traj, self.topology, self.Rmin, self.Rmax)
        elif partition_scheme == 'hysteretic':
            wrapper = HystereticPartition(self.traj, self.topology, self.Rmin_qm, self.Rmin, self.Rmin_bf, self.Rmax)
        else:
            raise ValueError("{} partition not implemented at this time".format(partition_scheme))

        return wrapper
        
                    
    def find_buffer_zone(self):

        if (self.run_ID == 0 or self.buffer_wrapper.class_type == 'distance'):
            qm = {}
            bf = {}
        else:
            qm =  self.systems[self.run_ID-1]['qm'].qm_residues
            bf =  self.systems[self.run_ID-1]['qm'].buffer_groups

        self.buffer_wrapper.define_buffer_zone(self.qm_center, self.qm_center_residues, prev_qm=qm, prev_bf=bf)

        self.qm_atoms = self.buffer_wrapper.get_qm_atoms()
        self.qm_residues = self.buffer_wrapper.get_qm_residues()
        self.qm_center_weight_ratio = self.buffer_wrapper.get_qm_center_info() 
        self.buffer_groups = self.buffer_wrapper.get_buffer_groups()

        # getting information for buffer groups
        self.buffer_distance = {}
        for i, buf in self.buffer_groups.items():
            self.buffer_distance[i] = buf.r_i
            buf.s_i, buf.d_s_i = self.compute_lamda_i(buf.r_i)

    @abstractmethod
    def find_configurations(self, info):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def run_aqmmm(self):
        """
        Function implemented in individual child classes
        """
        pass



