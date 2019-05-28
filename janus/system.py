import numpy as np
from copy import deepcopy
from mendeleev import element

class System(object):
    """
    A class that stores system information.
    Holds information such as energy, forces, and positions for 
    a QM/MM partition.
    Stores qmmm and aqmmm information as well.

    Parameters
    ----------
    qm indices : list 
        indices of the atoms of the QM region
    run_ID : int 
        the current step of the MD simulation
    partition_ID : int 
        An identifier for the specfic partition in aqmmm computations, default is 'qm'
    """

    def __init__(self, qm_indices, qm_residues, run_ID, partition_ID='qm'):

        self.qm_atoms = deepcopy(qm_indices)
        self.qm_residues = deepcopy(qm_residues)
        self.run_ID = run_ID
        self.partition_ID = partition_ID
        self.qm_positions = None
        self.buffer_groups = None
        self.switching_functions = None
        self.qmmm_forces = None
        self.entire_sys = {}
        self.primary_subsys = {}
        self.second_subsys = {}
        self.boundary = {}
        self.zero_energy = 0.0
        self.qmmm_energy = 0.0
        self.aqmmm_energy= 1.0

    def compute_scale_factor_g(qm, mm, link):
        """
        Computes scale factor g for link atom, RC, and RCD schemes. 
        The equation used to compute g is:
        
        .. math::
            \frac{R_{qm} + R_{link}}{R_{qm} + R_{mm}}
        
        where R is the pyykko covalent radius of an atom.

        Parameters
        ----------
        qm : str
            element symbol of the QM atom involved in broken bond 
        mm : str
            element symbol of the MM atom involved in broken bond 
        link : str
            element symbol for link atom

        Returns
        -------
        float
            g, the scaling factor

        Examples
        --------
        >>> compute_scale_factor_g('C', 'C', 'H')

        """
        
        r_qm = element(qm).covalent_radius_pyykko
        r_mm = element(mm).covalent_radius_pyykko
        r_link = element(link).covalent_radius_pyykko

        g = (r_qm + r_link)/(r_qm + r_mm)
        
        return g

class Buffer(object):
    """
    A class to store information for buffer groups 
    from aqmmm computations. This includes the atom indicies contained
    in the buffer group, the switching function, COM coordinates, etc.

    Parameters
    ----------
    ID : int 
        the identifer for the buffer group
    """

    def __init__(self, ID):
        """
        Initializes buffer class

        
        """

        self.ID = ID
        self.atoms = []
        self.COM_coord = None
        self.atom_weights = None
        self.weight_ratio = None
        self.r_i = None
        self.r_i_vect = None
        self.s_i = None
        self.d_s_i = None


