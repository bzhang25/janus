from copy import deepcopy
import numpy as np
import mdtraj as md
from janus.system import System
import time

class QMMM(object):
    """
    QMMM class for QMMM computations

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
        qm_atoms : list
            Indicies that define the QM region. Only static in traditional QM/MM
        qmmm_scheme : str 
            Scheme for computing QM/MM energies and gradients, 
            only substractive available(default)
        embedding_method : str 
            Embedding method to use for QM/MM. 
            Mechanical(default) and Electrostatic available
        boundary_treatment : str 
            Method for treating dangling bonds in the QM region,
            link_atom(default), RC, and RCD available
        link_atom_element : str 
            Element to use for link atom, default is H. 
            Beware of using others (not all functionality tested)
        
    """

    def __init__(self, hl_wrapper, 
                       ll_wrapper, 
                       sys_info,
                       sys_info_format='pdb',
                       qm_atoms=[],
                       qmmm_scheme='subtractive', 
                       embedding_method='Mechanical', 
                       boundary_treatment='link_atom',
                       link_atom_element='H'):
        
        self.class_type = 'QMMM'
        self.hl_wrapper = hl_wrapper
        self.ll_wrapper = ll_wrapper
        self.qm_geometry = None
        self.run_ID = 0

        self.traj = self.convert_input(sys_info, sys_info_format)
        self.topology = self.traj.topology
        self.positions = self.traj.xyz[0]

        self.qm_atoms = qm_atoms
        self.qmmm_scheme = qmmm_scheme
        self.embedding_method = embedding_method
        self.boundary_treatment = boundary_treatment
        self.link_atom_element = link_atom_element
        self.time_zero_energy = 0

        self.systems = {}

    def run_qmmm(self, main_info, wrapper_type):
        """
        Drives QM/MM computation.
        Updates the positions and topology given in main_info,
        and determines the QM/MM energy and gradients

        Parameters
        ----------
        main_info : dict 
            Contains the energy, forces, topology, and position information 
            for the whole system
        wrapper_type : str
            Defines the program used to obtain main_info
        """

        self.update_traj(main_info['positions'], main_info['topology'], wrapper_type)

        system = System(qm_indices=self.qm_atoms, qm_residues=None, run_ID=self.run_ID)

        if self.embedding_method =='Mechanical':
            t_all = self.mechanical(system, main_info)
        elif self.embedding_method =='Electrostatic':
            t_all = self.electrostatic(system, main_info)
        else:
            print('only mechanical and electrostatic embedding schemes implemented at this time')
            
        self.systems[self.run_ID] = {}
        self.systems[self.run_ID][system.partition_ID] = system
        self.systems[self.run_ID]['qmmm_forces'] = system.qmmm_forces 
        self.systems[self.run_ID]['qmmm_energy'] = system.qmmm_energy 
        #self.systems[self.run_ID]['kinetic_energy'] = main_info['kinetic']
        self.systems[self.run_ID]['kinetic_energy'] = system.entire_sys['kinetic']

        #if self.run_ID % 10 == 0:
        print('!', self.run_ID, self.systems[self.run_ID]['qmmm_energy'] + self.systems[self.run_ID]['kinetic_energy'])
            # add kinetic in total qmmm_energy

        # updates current step count
        self.run_ID += 1
        
        # delete the information of 2 runs before, only save current run and previous run information at a time
        if self.run_ID > 1:
            del self.systems[self.run_ID - 2]

        return t_all

    def update_traj(self, position, topology, wrapper_type):
        """
        Updates the positions and topology of self.traj,
        a MDtraj trajectory object

        Parameters
        ----------
        positions : list 
            positions in nm
        topology : OpenMM topology object
            If the mm program is  OpenMM
        wrapper_type : str
            Defines the program used to obtain topology and positions

       """ 
        # later can think about saving instead of making new instance
        # convert openmm topology to mdtraj topology
        if wrapper_type == 'OpenMM':
            top = md.Topology.from_openmm(topology)
        for atom in top.atoms:
            atom.serial = atom.index + 1

        self.traj = md.Trajectory(position, top)
        self.topology = self.traj.topology
        self.positions = self.traj.xyz[0]


    def mechanical(self, system, main_info):
        """
        Gets energies of needed components and computes
        a QM/MM energy with a subtractive mechanical embedding scheme
        using the formula

        E(QM/MM) = E(MM)_entire_sys - E(MM)_primary_subsys + E(QM)_primary_subsys

        Parameters
        ----------
        system : :class:`~janus.system.System`
            The system in which to save qmmm energy and forces
        main_info : dict 
            contains the energy and forces for the whole system

        """

        if self.qmmm_scheme == 'subtractive':
            # Get MM energy on whole system
            t_md1 = time.time()
            system.entire_sys = self.ll_wrapper.get_energy_and_gradient(self.traj)
            t_md2 = time.time()
            print('entire', system.entire_sys['energy'])

            #print(system.entire_sys['energy'])

            # Get MM energy on QM region
            print('calling make primary subsys trajectory')
            traj_ps, link_indices = self.make_primary_subsys_trajectory(qm_atoms=system.qm_atoms)
            system.primary_subsys['trajectory'] = traj_ps
            print('getting mm energy and gradient of qm region')
            t_md3 = time.time()
            system.primary_subsys['ll'] = self.ll_wrapper.get_energy_and_gradient(traj_ps, include_coulomb='no_link', link_atoms=link_indices)
            t_md4 = time.time()
            #print('ll', system.primary_subsys['ll']['energy'])

            # Get QM energy
            print('getting qm energy and gradient of qm region')
            t_qm1 = time.time()
            system.primary_subsys['hl'] = self.hl_wrapper.get_energy_and_gradient(traj_ps)
            t_qm2 = time.time()
            #print('hl', system.primary_subsys['hl']['energy'])
            #print('hl', system.primary_subsys['hl']['gradients'])

            # Compute the total QM/MM energy based on
            # subtractive Mechanical embedding
            system.qmmm_energy = system.entire_sys['energy']\
                        - system.primary_subsys['ll']['energy']\
                        + system.primary_subsys['hl']['energy']

            self.compute_gradients(system)
        else:
            print('only a subtractive scheme is implemented at this time')
        t_final = (t_md2 - t_md1) + (t_md4 - t_md3) + (t_qm2 - t_qm1)
        return t_final

    def electrostatic(self, system, main_info):
        """
        Gets energies of needed components and computes
        a QM/MM energy with a subtractive electrostatic embedding scheme

        E(QM/MM) = E(MM no coulomb)_entire_sys - E(MM no coulomb)_primary_subsys 
                 + E(QM)_primary_subsys + E(MM just coulomb)_secondary_subsys

        Parameters
        ----------
        system : :class:`~janus.system.System`
            The system in which to save qmmm energy and forces
        main_info : dict 
            contains the energy and forces for the whole system

        """ 

        if self.qmmm_scheme == 'subtractive':

            # Get MM energy on whole system
            t_md1 = time.time()
            system.entire_sys = self.ll_wrapper.get_energy_and_gradient(self.traj)
            t_md2 = time.time()
            print('entire', system.entire_sys['energy'])

            # Get MM energy on QM region
            traj_ps, link_indices = self.make_primary_subsys_trajectory(qm_atoms=system.qm_atoms)
            system.primary_subsys['trajectory'] = traj_ps
            t_md3 = time.time()
            system.primary_subsys['ll'] = self.ll_wrapper.get_energy_and_gradient(traj_ps, include_coulomb=None)
            t_md4 = time.time()

            # Get MM coulomb energy on secondary subsystem
            traj_ss = self.make_second_subsys_trajectory()
            system.second_subsys['trajectory'] = traj_ss
            t_md5 = time.time()
            system.second_subsys['ll'] = self.ll_wrapper.get_energy_and_gradient(traj_ss, include_coulomb='only')
            t_md6 = time.time()

            # Get QM energy
            charges, t_md7 = self.get_external_charges(system)
            t_qm1 = time.time()
            system.primary_subsys['hl'] = self.hl_wrapper.get_energy_and_gradient(traj_ps, charges=charges)
            t_qm2 = time.time()

            # Compute the total QM/MM energy based on
            # subtractive Mechanical embedding
            system.qmmm_energy = system.entire_sys['energy']\
                        - system.primary_subsys['ll']['energy']\
                        + system.second_subsys['ll']['energy']\
                        + system.primary_subsys['hl']['energy']

            self.compute_gradients(system)

        else:
            print('only a subtractive scheme is implemented at this time')
        t_final = (t_md2 - t_md1) + (t_md4 - t_md3) + (t_md6 - t_md5) + (t_qm2 - t_qm1) + t_md7
        return t_final

    def compute_gradients(self, system):
        """
        Computes the QM/MM gradients 

        Note
        ----
        RCD gradients currently not implemented

        Parameters
        ----------
        system : :class:`~janus.system.System`
            The system in which to save qmmm energy and forces

        """
        # NEED TO MAKE SURE: am I working with GRADIENTS or FORCES? NEED TO MAKE SURE CONSISTENT!
        # NEED TO MAKE SURE UNITS CONSISTENT

        if self.qmmm_scheme == 'subtractive':

            entire_grad = system.entire_sys['gradients']
            ps_mm_grad, qm_grad = system.primary_subsys['ll']['gradients'], system.primary_subsys['hl']['gradients']
            #print('ps_mm', ps_mm_grad)
            #print('qm', qm_grad)
            qmmm_force = {}
            print('!! sys qm_atom', system.qm_atoms)
                
            # iterate over list of qm atoms
            for i, atom in enumerate(system.qm_atoms):

                # compute the qmmm gradient for the qm atoms: 
                # mm_entire - mm_primary + qm
                # multiply by -1 to get from gradients to forces
                # these are in units of au_bohr, convert to openmm units in openmm wrapper
                #qmmm_force[atom] = -1 * (entire_grad[atom] - ps_mm_grad[i] + qm_grad[i])
                #print(atom, 'mm ps', -1*ps_mm_grad[i]*self.ll_wrapper.au_bohr_to_kjmol_nm)
                #print(atom, 'qm ps', -1*qm_grad[i]*self.ll_wrapper.au_bohr_to_kjmol_nm)
                #print(atom, 'entire', -1*entire_grad[atom]*self.ll_wrapper.au_bohr_to_kjmol_nm)
                qmmm_force[atom] = -1 * (- ps_mm_grad[i] + qm_grad[i])
                
                # treating gradients for link atoms
                if self.qmmm_boundary_bonds:
                    for j, link in self.link_atoms.items():
                        if isinstance(j, int):
                            q1 = link['qm_atom'].index
                            m1 = link['mm_atom'].index
                            link_index = link['link_atom_index']
                            g = link['scale_factor'] 
                            if atom == q1:
                                if self.boundary_treatment == 'link_atom':
                                    # Project forces of link atoms onto the mm and qm atoms of the link atom bond
                                    # need to make sure sign is correct
                                    qmmm_force[q1] += -(1 - g) * ps_mm_grad[link_index] + (1 - g) * qm_grad[link_index]
                                    qmmm_force[m1] = -g * ps_mm_grad[link_index] + g * qm_grad[link_index]
                                    
                            # # Forces on M2 requires forces on point charges which I'm not sure about so need to double check
                            # if self.boundary_treatment == 'RC' or self.boundary_treatment == 'RCD':
                            #     qmmm_force[atom] += -(1 - g) * ps_mm_grad[-1] + (1 - g) * qm_grad[-1]
                            #     qmmm_force[m1] += -g * ps_mm_grad[-1] + g * qm_grad[-1]

            # MM atoms
           # for a in range(len(entire_grad)):
           #     if a not in system.qm_atoms:
           #         qmmm_force[a] = -1 * entire_grad[a]


            if 'll' in system.second_subsys:
                # iterate over list of mm atoms
                for i, atom in enumerate(self.mm_atoms):
                    qmmm_force[atom] = -1 * system.second_subsys['ll']['gradients'][i]

            system.qmmm_forces = qmmm_force
        

    def find_boundary_bonds(self, qm_atoms=None):
        """
        Identified any covalent bonds that the QM/MM boundary cuts across

        Parameters
        ----------
        qm_atoms : list 
            Indicies corresponding to the atoms in
            the primary subsystem. Default is None and uses self.qm_atoms

        Examples
        --------
        >>> find_boundary_bonds()
        find_boundary_bonds(qm_atoms=[0,1,2,3])
        """

        if qm_atoms is None:
            qm_atoms = self.qm_atoms

        self.qmmm_boundary_bonds = []
        # determining if there are bonds that need to be cut
        for bond in self.topology.bonds:
            # find any bonds that involve the qm atoms
            # iterates over tuple of mdtraj atom objects
            if (bond[0].index in qm_atoms or bond[1].index in qm_atoms):
                # isolate bonds that involve one in the qm atoms and one outside
                if (bond[0].index not in qm_atoms or bond[1].index not in qm_atoms):
                    qm_atom = {}
                    mm_atom = {}
                    if bond[0].index in qm_atoms:
                        qm_atom = bond[0]
                        mm_atom = bond[1]
                    else:
                        qm_atom = bond[1]
                        mm_atom = bond[0]
                    self.qmmm_boundary_bonds.append((qm_atom, mm_atom))


    def prepare_link_atom(self):
        """
        Identifies where to put link atom.
        Saves the qm and mm atom associated with the bond being cut across the QM/MM boundary
        and computes the g scaling factor for the link atom.
        """

        self.link_atoms = {}
        self.link_atoms['all_mm'] = []
        self.link_atoms['all_outer_bonds'] = []

        for i, bond in enumerate(self.qmmm_boundary_bonds):

            qm = bond[0]
            mm = bond[1]

            self.link_atoms[i] = {}

            self.link_atoms[i]['qm_atom'] = qm
            self.link_atoms[i]['mm_atom'] = mm
            self.link_atoms['all_mm'].append(mm.index)

            self.link_atoms[i]['link_atom'] = self.link_atom_element
            g = System.compute_scale_factor_g(qm.element.symbol, mm.element.symbol, self.link_atom_element)
            self.link_atoms[i]['scale_factor'] = g 
            # this is in nm
            self.link_atoms[i]['link_positions'] = (1-g) * self.positions[qm.index] + g*self.positions[mm.index]

            if self.boundary_treatment == 'RC' or self.boundary_treatment == 'RCD':
                bonds = []
                # find index of atoms bonded to mm atom
                for bond in self.topology.bonds:
                    if bond[0]== mm or bond[1] == mm:
                        if bond[0] != qm and bond[1] != qm:
                            if bond[0] != mm:
                                bonds.append(bond[0].index)
                            elif bond[1] != mm:
                                bonds.append(bond[1].index)

                self.link_atoms[i]['bonds_to_mm'] = bonds
                self.link_atoms['all_outer_bonds'].append(bonds)


    def make_primary_subsys_trajectory(self, qm_atoms=None):
        '''
        Creates a MDtraj trajectory object with just the 
        primary subsystem, and adds in any link atoms

        Note
        ----
        Currently adds just H as link atom, need to expand. Also, H is added
        as a very specific H1 atom, or else the connectivity in the create_new_residue_templates
        function in openmm gets messed up and gives a "set of atoms match but bonds are different
        error"

        Parameters
        ----------
        qm_atoms : list 
            atom indicies corresponding to the atoms in
            the primary subsystem. Default is None and uses self.qm_atoms

        Returns
        -------
        MDtraj trajectory object
        list
            The link atom indices in traj

        Examples
        --------
        >>> make_primary_subsys_trajectory([0,1,2])
        make_primary_subsys_trajectory()
        '''

        if qm_atoms is None:
            qm_atoms = self.qm_atoms
        
        print('number of qm_atoms fed into make primary trajectory', len(qm_atoms))

        self.find_boundary_bonds(qm_atoms)
        traj = self.traj.atom_slice(qm_atoms)

        link_indices = []
        if self.qmmm_boundary_bonds:
            self.prepare_link_atom()

            for i, link in self.link_atoms.items():
                if isinstance(i, int):
                
                    link_element = md.element.Element.getBySymbol(link['link_atom'])

                    for atom in traj.topology.atoms:
                        if atom.serial == link['qm_atom'].serial:
                            traj.topology.add_atom(name='H1', element=link_element, residue=atom.residue, serial='link')
                            for atom2 in traj.topology.atoms:
                                if atom2.serial == 'link':
                                    traj.topology.add_bond(atom2, atom)
                                    link['link_atom_index'] = atom2.index

                    link_indices.append(link['link_atom_index'])
                    traj.xyz = np.append(traj.xyz[0], [link['link_positions']], axis=0)
        
        return traj, link_indices

    def make_second_subsys_trajectory(self, qm_atoms=None):
        '''
        Creates a MDtraj trajectory object with just the 
        secondary subsystem

        Parameters
        ----------
        qm_atoms : list 
            atom indicies corresponding to the atoms in
            the primary subsystem. Default is None and uses self.qm_atoms

        Returns
        -------
        MDtraj trajectory object

        Examples
        --------
        >>> make_second_subsys_trajectory([0,1,2])
        make_second_subsys_trajectory()
        '''

        if qm_atoms is None:
            qm_atoms = self.qm_atoms
    
        self.mm_atoms = [i for i in range(self.traj.n_atoms) if i not in qm_atoms]

        traj = self.traj.atom_slice(self.mm_atoms)

        return traj


    def get_forces(self, run_ID=None):
        """
        Function to return qmmm forces

        Parameters
        ----------
        run_ID : int
            identifies which step to get forces from
        
        Returns
        -------
        dict
            qmmm forces in au/bohr

        Examples
        --------
        >>> forces = get_forces()
        """
        if run_ID is None:
            run_ID = self.run_ID - 1

        return self.systems[run_ID]['qmmm_forces']


    def get_external_charges(self, system):
        #TODO: at some point maybe need to migrate this to mm_wrapper
        """
        Gets the point charges of atoms from secondary subsystem for electrostatic embedding 

        Parameters
        ----------
        system : :class:`~janus.system.System`
            The system in which to save qmmm energy and forces

        Returns
        -------
        list
            charges and corresponding positions in angstroms as xyz coordinates

        """
        charges = []
        # in angstroms
        es_pos = 10*system.entire_sys['positions']
        t_md1 = time.time()
        charge = self.ll_wrapper.get_main_charges()
        t_md2 = time.time()

        if self.embedding_method == 'Mechanical':
            return None

        elif self.boundary_treatment == 'link_atom':
            for i, chrg in enumerate(charge):
                # add every atom not in qm system 
                if i not in system.qm_atoms:
                    # save positions in angstroms
                    charges.append([chrg, es_pos[i][0], es_pos[i][1], es_pos[i][2]])
        
        # This is for the RC and RCD schemes
        elif self.boundary_treatment == 'RC':

            for i, chrg in enumerate(charge):
                # add every atom not in qm system or the M1 atom 
                if i not in system.qm_atoms and i not in self.link_atoms['all_mm']:
                        charges.append([chrg, es_pos[i][0], es_pos[i][1], es_pos[i][2]])
            
            bonds = self.link_atoms['all_outer_bonds']

            for i, index in enumerate(self.link_atoms['all_mm']):

                # check to see that the M1 atom is attached to any M2 atoms
                if bonds[i]:
                    # get q0
                    q0 = charge[index] / len(bonds[i])

                    # get positions in angstroms
                    positions = self.get_redistributed_positions(es_pos, bonds[i], index)

                    for pos in positions:
                        charges.append([q0, pos[0], pos[1], pos[2]])

        elif self.boundary_treatment == 'RCD':

            bonds = self.link_atoms['all_outer_bonds']
            m1_m2 = []

            for i, index in enumerate(self.link_atoms['all_mm']):

                m1_m2.append(index)
                # get q0

                if bonds[i]:

                    q0 = charge[index] / len(bonds[i])
                    q0_RCD = q0 * 2

                    # get positions in angstroms
                    positions = self.get_redistributed_positions(es_pos, bonds[i], index)

                    for pos in positions:
                        charges.append([q0_RCD, pos[0], pos[1], pos[2]])

                    for bond in bonds[i]:
                        m1_m2.append(bond)
                        charges.append([charge[bond] - q0, es_pos[bond][0], es_pos[bond][1], es_pos[bond][2]])

            for i, chrg in enumerate(charge):
                # add every atom not in qm system or the M1 atom 
                if i not in system.qm_atoms and i not in m1_m2:
                        charges.append([chrg, es_pos[i][0], es_pos[i][1], es_pos[i][2]])

                
        return charges, t_md2 - t_md1

    def get_redistributed_positions(self, positions, bonds, mm):
        """
        Gets the positions for the redistributed point charges in the RC and RCD schemes

        Parameters
        ----------
        positions : list 
        bonds : list 
            indices of all atoms (in secondary subsystem) bonded to M1  
        mm : int 
            the index of M1

        Returns
        -------
        list
            positions for the redistributed charges

        """
        
        pos = []
    
        for bond in bonds:
            new_pos = ((positions[bond] + positions[mm]) / 2)
            pos.append(new_pos)
        
        return pos

            
    def convert_input(self, fil, form):
        """
        Converts a set of input files into a MD trajectory

        Parameters
        ----------
        fil : str 
            A string with the filename or a list with multiple filenames 
            that contain position and topology information. 
        form : str 
            Describes what kind of input is contained in sys_info. Default is pdb.

        Returns
        -------
        MDtraj object
        
        """
            
        print(form)
        if form == 'pdb':
            traj = md.load(fil)

        if form == 'Amber':

            for f in fil:
                if f.endswith('prmtop'):
                    top_fil = f
                    use_pdb = False
                if f.endswith('pdb'):
                    pdb_fil = f
                    use_pdb = True
                if f.endswith('inpcrd'):
                    crd_fil = f
            print('loading') 
            print(pdb_fil)
            if use_pdb is True:
                traj = md.load(pdb_fil)
            else:
                traj = md.load(crd_fil, top=top_fil)

        return traj
