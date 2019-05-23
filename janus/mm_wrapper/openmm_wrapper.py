import simtk.openmm.app as OM_app
import simtk.openmm as OM
import simtk.unit as OM_unit
from mdtraj.reporters import NetCDFReporter
from janus.mm_wrapper import MMWrapper
import numpy as np
import pickle
from copy import deepcopy

class OpenMMWrapper(MMWrapper):
    """
    A MM wrapper class that calls OpenMM
    to obtain molecular mechanics information. 
    Also can be used to take steps in a molecular dynamics simulation. 
    Class inherits from MMWrapper.

    Parameters
    ----------
    sys_info : str 
        A string with the filename or a list with multiple filenames 
        that contain position and topology information. Default is None.
    sys_info_format : str 
        A str describing what kind of input is contained in sys_info. Default is pdb.
        Possible values also include Amber and Gromacs.
    mm_forcefield : str
        The name of the forcefield to use, default is amber99sb.xml.
    mm_water_forcefield : str
        The name of the forcefield to use for water, default is tip3p.xml.
    NVE_integrator : str
        What type of integrator to use for a NVE ensemble, default is Verlet.
    NVT_integrator : str
        What type of integrator to use for a NVT ensemble, default is Langevin.
    temp : Int
        The temperature at which to run a simulation in kelvin, default is 300.
    step_size : float
        The step size to integrate system in femtoseconds, default is 1.
    fric_coeff : float
        friction coefficient to couple the system to heat bath in a NVT ensemble in inverse
        picoseconds, default is 1.
    nonbondedCutoff : float 
        The cutoff distance for nonbonded interactions in nanometers,
        default is 1.
    **kwargs : dict
        Other parameters for OpenMM, which include:
        - nonbondedMethod : method for nonbonded interactions, default is OM_app.NoCutoff
        - constraints : which bonds and angles implemented with constraints,
                        default is None
        - rigid_water : whether water is treated as rigid, default is True
        - removeCMMotion : whether to include a CMMotionRemover, default is True
        - ignoreExternalBonds : whether to ignore external bonds when matching residues to templates,
                                default is True
        - flexibleConstraints : whether to add parameters for constrained parameters,
                                default is False
        - hydrogenMass : the mass to use for hydrogen atoms bonded to heavy atoms,
                        default is False
        - residueTemplates : allows user to specify a template for a residue,
                            default is empty dict {}
        - switchDistance : the distance to turn on potential energy switching function for 
                            Lennard-Jones interactions. Default is None
        - keywords for MD simulation parameters.
            For possible keywords consult the Molecular Dynamics section of the manual. 

        Each pair of key:value in the dictionary is given as a string.
        For more information about these pararmeters and 
        other possible parameter values consult docs.openmm.org

    """

    def __init__(self, sys_info=None, 
                       sys_info_format='pdb', 
                       mm_forcefield='amber99sb.xml',
                       mm_water_forcefield='tip3p.xml',
                       NVE_integrator='Verlet',
                       NVT_integrator='Langevin',
                       temp = 300,
                       step_size = 1,
                       fric_coeff = 1,
                       nonbondedCutoff=0.8,
                       **kwargs):

        super().__init__(class_type="OpenMM",
                         sys_info=sys_info,
                         sys_info_format=sys_info_format)

        self.ff = mm_forcefield
        self.ff_water = mm_water_forcefield
        self.NVE_integrator = NVE_integrator
        self.NVT_integrator = NVT_integrator
        self.temp = temp*OM_unit.kelvin
        self.step_size = step_size*OM_unit.femtoseconds
        self.fric_coeff = fric_coeff/OM_unit.picosecond
        self.nonbondedCutoff=nonbondedCutoff*OM_unit.nanometer

        self.nonbondedMethod = OM_app.NoCutoff
        self.constraints = None
        self.hydrogenMass = None
        self.switchDistance = None
        self.residueTemplates={}
        self.rigid_water=False
        self.removeCMMotion=True
        self.flexibleConstraints=False 
        self.ignoreExternalBonds=True

        openmm_param = ['nonbondedMethod', 'constraints', 'hydrogenMass', 'switchDistance', 'residueTemplates']

        for k, v in kwargs.items():
            if k in openmm_param:
                setattr(self, k, eval(v))
            else:
                setattr(self, k, v)

        self.post_processing_input()

        if self.md_ensemble == 'NVT':
            self.integrator = self.NVT_integrator
        elif self.md_ensemble == 'NVE':
            self.integrator = self.NVE_integrator

        self.positions = None

        self.convert_input()

    def initialize(self, embedding_method):
        """
        Gets information for the system
        of interest in its initial state.
        The simulation object and information dictionary returned 
        by :func:`~janus.mm_wrapper.OpenMMWrapper.compute_info` is saved.
        
        Parameters
        ----------
        embedding_method : str
            what QM/MM embedding method to use for initialization.
            If 'Mechanical', all forces are included
            If 'Electrostatic', all coulombic forces are excluded
        """

        # should I minimize energy here? If so, need to return new positions

        print('initializing mm wrapper')
        if (self.other_md_ensembles is not None and self.other_ensemble_steps is not None):
            print('other ensembles')
            print(self.other_md_ensembles)
            for i, ensemble in enumerate(self.other_md_ensembles):
                print('running equilibrating ensemble {} for {} steps'.format(ensemble,self.other_ensemble_steps[i]))
                
                if ensemble == 'NVT':
                    integrator = self.NVT_integrator
                elif ensemble == 'NVE':
                    integrator = self.NVE_integrator

                OM_system = self.create_openmm_system(self.topology)
                simulation, integrator_obj = self.create_openmm_simulation(OM_system, self.topology, self.positions, integrator, return_integrator=True)
                simulation.minimizeEnergy()

                #simulation.reporters.append(NetCDFReporter('output_nvt.nc', 50))
                #simulation.reporters.append(OM_app.StateDataReporter('info_nvt.dat', 100, step=True,
                #potentialEnergy=True, kineticEnergy=True, totalEnergy=True,temperature=True))

                simulation.step(self.other_ensemble_steps[i])

                state = simulation.context.getState(getPositions=True)
                pos = state.getPositions()
                # also not sure if there should be option for returning information about these
                # not sure if should save this
                del simulation, state, OM_system, integrator_obj
        else:
            pos = self.positions

        if embedding_method == 'Mechanical':
            self.main_simulation, self.main_info =\
            self.compute_info(self.topology, pos, initialize=True, return_simulation=True, minimize=False)

        elif embedding_method == 'Electrostatic':
            self.main_simulation, self.main_info =\
            self.compute_info(self.topology, pos, include_coulomb=None, initialize=True, return_simulation=True, minimize=False)
        else:
            print('only mechanical and electrostatic embedding schemes implemented at this time')

    def restart(self, embedding_method, chkpt_file, restart_forces):

        # ensure every computation has same periodic box vector parameters
        self.topology.setPeriodicBoxVectors(self.PeriodicBoxVector)

        if embedding_method == 'Mechanical':
            # Create an OpenMM system from an object's topology
            OM_system = self.create_openmm_system(self.topology, initialize=True)
        elif embedding_method == 'Electrostatic':
            OM_system = self.create_openmm_system(self.topology, include_coulomb=None, initialize=True)

        print(self.integrator)
        # Create an OpenMM simulation from the openmm system, topology, and positions.
        self.main_simulation = self.create_openmm_simulation(OM_system, self.topology, self.positions, self.integrator)

        with open(chkpt_file, 'rb') as f:
            self.main_simulation.context.loadCheckpoint(f.read())

        with open(restart_forces, 'rb') as force_file:
            force = pickle.load(force_file)

        
        #self.set_up_reporters(self.main_simulation)
        # Calls openmm wrapper to get information specified
        #self.main_info = OpenMMWrapper.get_state_info(self.main_simulation,
        #                              energy=True,
        #                              positions=True,
        #                              forces=True)
        #print('before loading forces')
        #print(self.main_info)

        self.update_forces(force, self.qmmm_force, self.main_simulation)
        
        self.set_up_reporters(self.main_simulation)
        ## Calls openmm wrapper to get information specified
        self.main_info = OpenMMWrapper.get_state_info(self.main_simulation,
                                      energy=True,
                                      positions=True,
                                      forces=True)
        #print('after loading forces')
        #print(self.main_info)

    def take_updated_step(self, force):
        """
        Updates the system with forces from qmmm 
        and takes a simulation step

        Parameters
        ----------
        force : dict 
            forces(particle index : forces) in au/bohr to 
            be updated in custom qmmm force and fed into simulation

        """

        self.update_forces(force, self.qmmm_force, self.main_simulation)
        self.main_simulation.step(1)                                             # take a step
        self.main_info = self.get_main_info()                                    # get the energy and gradients after step
        self.positions = self.main_info['positions']                             # get positions after step
    
        #print('main info after step')
        #print(self.main_info)

    def update_forces(self, forces, force_obj, simulation):
        """
        Updates a simulation with external forces 

        Parameters
        ----------
        force : dict 
            forces(particle index : forces) in au/bohr to 
            be updated in a force object  and fed into simulation
        force_obj : OpenMM Force object
            the force object to add the forces to. Can be built in or custom
        simulation : OpenMM simulation object
            where the forces are to be updated in
        
        """

        current = OpenMMWrapper.get_state_info(self.main_simulation,
                                                energy=False,
                                                positions=False,
                                                forces=True)
        #for f in range(len(current['forces'])):
        #    print(f, current['forces'][f])

        for f, coord in forces.items():
            coord *= MMWrapper.au_bohr_to_kjmol_nm             # convert this back to openmm units
            #print('fed in forces')
            #print(f, coord)
            #print('forces from before')
            #print(current['forces'][f])
            #coord -= current['forces'][f]
            #print('new forces')
            #print(coord)
            force_obj.setParticleParameters(f, f, coord)  # need to figure out if the first 2 parameters always the same or not
            

        force_obj.updateParametersInContext(simulation.context)  # update forces with qmmm force
        #current2 = OpenMMWrapper.get_state_info(self.main_simulation,
        #                                        energy=False,
        #                                        positions=False,
        #                                        forces=True)
        #for f in range(len(current2['forces'])):
        #    print(f, current2['forces'][f])

    def take_step(self, num):
        """
        Takes a specified num of steps in the MD simulation
        
        Parameters
        ----------
        num : int
            the number of pure MD steps to take
        """

        if num != 0:
            self.main_simulation.step(num)

    def get_main_info(self):
        """
        Gets the information for the system of interest.
        Calls :func:`~janus.mm_wrapper.OpenMMWrapper.get_state_info`
        to obtain information.

        Returns
        -------
        dict
            Information including the current energy, gradient, 
            and positions for the system of interest
    
        """
        
        return OpenMMWrapper.get_state_info(self.main_simulation, main_info=True)

    def compute_info(self, topology, positions, include_coulomb='all', initialize=False, return_system=False, return_simulation=False, link_atoms=None, minimize=False):
        """
        Gets information about a system. 

        Parameters
        ----------
        topology : OpenMM topology object
        positions : OpenMM Vec3 vector 
            contains the positions of the system in nm
        include_coulomb : str
            whether to include coulombic interactions. 
            'all' (default) includes coulombic forces for all particles,
            'no_link' excludes coulombic forces for link atoms,
            'only' excludes all other forces for all atoms,
            'none' excludes coulombic forces for all particles.
        initialize : bool 
            Whether the main system is being initialized.
        return_system : bool 
            True(default) to return OpenMM system object
        return_simulation : bool 
            True(default) to return OpenMM simulation object
        link_atoms : list
            if included as a list with include_coulomb='no_link', specifies which 
            atoms to remove coulombic forces from. Default is None.
        minimize : bool
            whether to minimize the energy of the system

        Returns
        -------
        dict
            A dictionary with state information
        OpenMM system object
            OpenMM system object returned unless return_system=False
        OpenMM simulation object 
            OpenMM simulation object returned unless return_simulation=False

        Examples
        --------
        >>> system, simulation, state = compute_info(top, pos)

        >>> state = compute_info(top, pos, return_simulation=False, return_system=False)
        """

        # ensure every computation has same periodic box vector parameters
        topology.setPeriodicBoxVectors(self.PeriodicBoxVector)
        # Create an OpenMM system from an object's topology
        print('topology going into system')
        print(topology.getNumAtoms())
        OM_system = self.create_openmm_system(topology, include_coulomb, link_atoms,initialize=initialize)

        # Create an OpenMM simulation from the openmm system, topology, and positions.
        simulation = self.create_openmm_simulation(OM_system, topology, positions, self.integrator)

        if minimize is True:
            simulation.minimizeEnergy()

        if initialize is True:
        # set up reporters
            self.set_up_reporters(simulation) 

        # Calls openmm wrapper to get information specified
        state = OpenMMWrapper.get_state_info(simulation,
                                      energy=True,
                                      positions=True,
                                      forces=True)

        if return_system is True and return_simulation is True:
            return OM_system, simulation, state
        elif return_system is True and return_simulation is False:
            return OM_system, state
        elif return_system is False and return_simulation is True:
            return simulation, state
        else:
            return state


    def create_openmm_system(self, topology, include_coulomb='all', link_atoms=None, initialize=False):
        """
        Calls OpenMM to create an OpenMM System object give a topology,
        forcefield, and other parameters specified in the instantiation parameters.

        Note
        ----
        Currently there is no support for customized forcefields

        Parameters
        ----------
        topology : OpenMM topology object
        include_coulomb : str

            whether to include coulombic interactions. 
            'all' (default) includes coulombic forces for all particles,
            'no_link' excludes coulombic forces for link atoms,
            'only' excludes all other forces for all atoms,
            'none' excludes coulombic forces for all particles.

        link_atoms : list
            if included as a list with include_coulomb='no_link', specifies which 
            atoms to remove coulombic forces from. Default is None.
        initialize : bool 
            Whether the main system is being initialized.

        Returns
        -------
        OpenMM system object

        Examples
        --------
        >>> openmm_sys = create_openmm_system(topology)
        openmm_sys = create_openmm_system(top, include_coulomb='no_link', link_atoms=[0,1,2])
        openmm_sys = create_openmm_system(top, include_coulomb='only')
        openmm_sys = create_openmm_system(top, initialize=True)
        """

        # check to see if there are unmatched residues in pdb, create residue templates if there are
        if (self.system_info_format == 'pdb' or self.use_pdb is True):
            unmatched = self.forcefield.getUnmatchedResidues(topology)
            #if unmatched:
            #    self.create_new_residue_template(topology)
            print("number of residues for problem topology with unmatched residue")
            print(len(unmatched))
            if len(unmatched) > 0:
                print(unmatched[0])
                for atom in unmatched[0].atoms():
                    print(atom)
                for bonds in unmatched[0].bonds():
                    print(bonds)

            openmm_system = self.forcefield.createSystem(topology,
                                            nonbondedMethod=self.nonbondedMethod,
                                            constraints=self.constraints,
                                            hydrogenMass=self.hydrogenMass,
                                            switchDistance=self.switchDistance,
                                            residueTemplates=self.residueTemplates,
                                            nonbondedCutoff=self.nonbondedCutoff,
                                            rigidWater=self.rigid_water,
                                            removeCMMotion=self.removeCMMotion,
                                            flexibleConstraints=self.flexibleConstraints,
                                            ignoreExternalBonds=self.ignoreExternalBonds)

        elif (self.system_info_format == 'Amber' and self.use_pdb is False):

            if topology.getNumAtoms() != self.topology.getNumAtoms():
                print('reading topology not the same')
                forcefield = deepcopy(self.forcefield)
                forcefield.topology = topology
                print(forcefield.topology.getNumAtoms())
            else:
                print('reading topology same')
                forcefield = self.forcefield

            openmm_system = forcefield.createSystem(nonbondedMethod=self.nonbondedMethod,
                                            constraints=self.constraints,
                                            hydrogenMass=self.hydrogenMass,
                                            switchDistance=self.switchDistance,
                                            nonbondedCutoff=self.nonbondedCutoff,
                                            rigidWater=self.rigid_water,
                                            removeCMMotion=self.removeCMMotion)
            print('new system particles')
            print(openmm_system.getNumParticles())

        if initialize is True:                                             # this is for the initialization of the entire system
            self.qmmm_force = OM.CustomExternalForce("-x*fx-y*fy-z*fz")    # define a custom force for adding qmmm gradients
            self.qmmm_force.addPerParticleParameter('fx')
            self.qmmm_force.addPerParticleParameter('fy')
            self.qmmm_force.addPerParticleParameter('fz')
            
            for i in range(openmm_system.getNumParticles()):
                self.qmmm_force.addParticle(i, np.array([0.0, 0.0, 0.0]))
            
            openmm_system.addForce(self.qmmm_force)

            self.main_charges = [openmm_system.getForce(3).getParticleParameters(i)[0]/OM_unit.elementary_charge for i in range(openmm_system.getNumParticles())]

        # If in electrostatic embedding scheme need to get a system without coulombic interactions
        if include_coulomb == 'none':
            # get the nonbonded force
            self.set_charge_zero(openmm_system)

        if (include_coulomb == 'no_link' and link_atoms is not None):
            self.set_charge_zero(openmm_system, link_atoms)

        if include_coulomb == 'only':
        # Remove Bond, Angle, and Torsion forces to leave only nonbonded forces
            for i in range(openmm_system.getNumForces()):             
                if type(openmm_system.getForce(0)) is not OM.NonbondedForce:     
                    openmm_system.removeForce(0)                              
            self.set_LJ_zero(openmm_system)


        return openmm_system

    def set_charge_zero(self, OM_system, link_atoms=None):
        """
        Removes the coulombic forces by setting charges of 
        specified atoms to zero

        Parameters
        ----------
        OM_system : OpenMM system object
        link_atoms : list 
            link_atoms to set the charge to zero,
            if link_atoms is None (default), the charge of 
            all particles in the system will be set to zero
    
        Examples
        --------
        >>> set_charge_zero(system)
        set_charge_zero(system, link_atoms=[0,1,2])
        """

        for force in OM_system.getForces():
            if type(force) is OM.NonbondedForce:
                if link_atoms:
                    # set the charge of link atoms to 0 so the coulomb energy is zero
                    for i in link_atoms:
                        a = force.getParticleParameters(i)
                        force.setParticleParameters(i, charge=0.0, sigma=a[1], epsilon=a[2])
                else:
                    # set the charge of all particles to 0 so the coulomb energy is zero
                    for i in range(force.getNumParticles()):
                        a = force.getParticleParameters(i)
                        force.setParticleParameters(i, charge=0.0, sigma=a[1], epsilon=a[2])

    def set_LJ_zero(self, OM_system):
        """
        Removes the Lennard-Jones (van der Waals) force from the system

        Parameters
        ----------
        OM_system : OpenMM system object
    
        """

        for force in OM_system.getForces():
            if type(force) is OM.NonbondedForce:
                for i in range(force.getNumParticles()):
                    a = force.getParticleParameters(i)
                    force.setParticleParameters(i, charge=a[0], sigma=0.0, epsilon=0.0)
        

    def create_new_residue_template(self, topology):
        """
        Create a new OpeMM residue template when there is no matching residue 
        and registers it into self.forcefield forcefield object.
    
        Note
        ----
        currently, if there is unmatched name, currently only checks original 
        unmodified residue, N-terminus form, and C-terminus form. 
        This may not be robust.

        Parameters
        ----------
        topology : OpenMM topology object

        Examples
        --------
        >>> create_new_residue_template(topology)
        """
        template, unmatched_res = self.forcefield.generateTemplatesForUnmatchedResidues(topology)

        # Loop through list of unmatched residues
        print('Loop through list of unmatched residues')
        for i, res in enumerate(unmatched_res):
            res_name = res.name                             # get the name of the original unmodifed residue
            n_res_name = 'N' + res.name                     # get the name of the N-terminus form of original residue
            c_res_name = 'C' + res.name                     # get the name of the C-terminus form of original residue
            name = 'Modified_' + res_name                   # assign new name
            template[i].name = name

            # loop through all atoms in modified template and all atoms in orignal template to assign atom type
            print('loop through all atoms in modified template and all atoms in orignal template to assign atom type')
            for atom in template[i].atoms:
                for atom2 in self.forcefield._templates[res_name].atoms:
                    if atom.name == atom2.name:
                        atom.type = atom2.type
                # the following is for when there is a unmatched name, check the N and C terminus residues
                if atom.type == None:
                    print('check n')
                    for atom3 in self.forcefield._templates[n_res_name].atoms:
                        if atom.name == atom3.name:
                            atom.type = atom3.type
                if atom.type == None:
                    print('check c')
                    for atom4 in self.forcefield._templates[c_res_name].atoms:
                        if atom.name == atom4.name:
                            atom.type = atom4.type

            # override existing modified residues with same name
            if name in self.forcefield._templates:
                print('override existing modified residues with name {}'.format(name))
                template[i].overrideLevel = self.forcefield._templates[name].overrideLevel + 1

            # register the new template to the forcefield object
            print('register the new template to the forcefield object')
            self.forcefield.registerResidueTemplate(template[i])


    def create_openmm_simulation(self, openmm_system, topology, positions, integrator,  return_integrator=False, seed=0):
        """
        Creates an OpenMM simulation object given
        an OpenMM system, topology, and positions

        Parameters
        ----------
        openmm_system : OpenMM system object
        topology : OpenMM topology object
        positions : OpenMM Vec3 vector 
            contains the positions of the system in nm
        integrator : str
            What type of integrator to use. Currently support Langevin and Verlet.
        return_integrator : bool
            Whether to return the OpenMM integrator object
        seed : int
            Set a random seed number for the Langevin integrator. 
            Default is 0, which means seed is randomized every time.

        Returns
        -------
        OpenMM simulation object
        
        OpenMM integrator object
            Returned unless return_integrator is False

        Examples
        --------
        >>> create_open_simulation(openmm_sys, top, pos) 
        create_open_simulation(openmm_sys, pdb.topology. pdb.positions)
        """

        #print('using {} integrator'.format(integrator))
        if integrator == 'Langevin':
            integrator_obj = OM.LangevinIntegrator(self.temp, self.fric_coeff, self.step_size)
            integrator_obj.setRandomNumberSeed(seed)
        elif integrator == 'Verlet':
            integrator_obj = OM.VerletIntegrator(self.step_size)
        else:
            print('only Langevin integrator supported currently')

        simulation = OM_app.Simulation(topology, openmm_system, integrator_obj)
        simulation.context.setPositions(positions)

        if integrator == 'Verlet':
            simulation.context.setVelocitiesToTemperature(self.temp)

        if return_integrator is False:
            return simulation
        else:
            return simulation, integrator_obj

    def get_state_info(simulation,
                       main_info=False,
                       energy=True,
                       positions=True,
                       velocity=False,
                       forces=True,
                       parameters=False,
                       param_deriv=False,
                       periodic_box=False,
                       groups_included=-1):
        """
        Gets information like the kinetic and potential energy,
        positions, forces, and topology from an OpenMM state.
        Some of these may need to be made accessible to user.

        Parameters
        ----------
        simulation : OpenMM simulation object
        main_info : bool 
            specifies whether to return the topology of the system
        energy : bool 
            spcifies whether to get the energy, returned in hartrees(a.u.), default is true.
        positions : bool 
            specifies whether to get the positions, returns in nanometers, default is true
        velocity : bool 
            specifies whether to get the velocities, default is false
        forces : bool 
            specifies whether to get the forces acting on the system, returns as numpy array in jk/mol/nm, 
            as well as the gradients, in au/bohr, default is true
        parameters : bool 
            specifies whether to get the parameters of the state, default is false
        param_deriv : bool 
            specifies whether to get the parameter derivatives of the state, default is false
        periodic_box : bool 
            whether to translate the positions so the center of every molecule lies in the same periodic box, default is false
        groups : list
            a set of indices for which force groups to include when computing forces and energies. Default is all groups

        Returns
        -------
        dict
            Information specified by parameters.
            Keys include 'energy', 'potential', 'kinetic', 'forces',
            'gradients', 'topology'

        Examples
        --------
        >>> get_state_info(sim)
        get_state_info(sim, groups_included=set{0,1,2})
        get_state_info(sim, positions=True, forces=True)
        """

        state = simulation.context.getState(getEnergy=energy,
                                            getPositions=positions,
                                            getVelocities=velocity,
                                            getForces=forces,
                                            getParameters=parameters,
                                            getParameterDerivatives=param_deriv,
                                            enforcePeriodicBox=periodic_box,
                                            groups=groups_included)

        values = {}
        # divide by unit to give value without units, then convert value to atomic units
        if energy is True:
            values['potential'] = state.getPotentialEnergy()/OM_unit.kilojoule_per_mole
            values['potential'] *= MMWrapper.kjmol_to_au
            values['kinetic'] = state.getKineticEnergy()/OM_unit.kilojoule_per_mole
            values['kinetic'] *= MMWrapper.kjmol_to_au
            values['energy'] = values['potential'] 

        if positions is True:
            values['positions'] = state.getPositions(asNumpy=True)/OM_unit.nanometer

        if forces is True:
            values['forces'] = state.getForces(asNumpy=True)/(OM_unit.kilojoule_per_mole/OM_unit.nanometer)
            values['gradients'] = (-1) * values['forces'] * MMWrapper.kjmol_nm_to_au_bohr   

        if main_info is True:
            # need to check if the topology actually updates 
            values['topology'] = simulation.topology

        return values

    def write_pdb(self, info):
        """
        Write a pdb file 

        Parameters
        ----------
        info : dict
            dictionary containing the topology and positions of 
            the system to write to file
             
        """

        if self.return_system is True: 
            OM_app.PDBFile.writeFile(info['topology'], info['positions'], open(self.return_system_filename, 'w'))
 

    def create_modeller(self, atoms, keep_atoms=False):
        """
        Makes a OpenMM modeller object based on given geometry

        Parameters
        ----------
        atoms : list
            The subset of atom indices to either keep or delete
        keep_atoms : bool 
            Whether to keep the atoms specified in the modeller or delete them.
            Default is false.

        Returns
        -------
        A OpenMM modeller object

        Examples
        --------
        >>> modeller = self.make_modeller()
        modeller = self.make_modeller(keep_qm=True)
        """

        modeller = OM_app.Modeller(self.topology, self.pdb.getPositions())
        if keep_atoms is False:
            OpenMMWrapper.delete_atoms(modeller, atoms)
        elif keep_atoms is True:
            OpenMMWrapper.keep_atoms(modeller, atoms)
        return modeller

    def keep_atoms(model, atoms):
            """
            Acts on an OpenMM Modeller object to keep the specified
            atoms in the MM system and deletes everything else

            Parameters
            ----------
            model : OpenMM Modeller object
            atoms : list 
                which atoms to keep in an OpenMM Modeller object

            Examples
            --------
            >>> keep_atom(mod, [0,1])
            keep_atom(mod, ['O', 'H'])
            """
            lis = []

            for atom in model.topology.atoms():
                if type(atoms[0]) is int:
                    if atom.index not in atoms:
                        lis.append(atom)
                elif type(atoms[0]) is str:
                    if atom.name not in atoms:
                        lis.append(atom)
            model.delete(lis)


    def delete_atoms(model, atoms):
         """
         Delete specified atoms from an OpenMM Modeller object

         Parameters
         ----------
         model : OpenMM Modeller object
         atoms : list 
            which atom IDs (int) or atom names (str) to delete
            an OpenMM Modeller object

         Examples
         --------
         >>> delete_atoms(model, [0, 3, 5])
         delete_atoms(model, ['Cl'])
         """
         lis = []
         for atom in model.topology.atoms():
             if type(atoms[0]) is int:
                 if atom.index in atoms:
                     lis.append(atom)
             elif type(atoms[0]) is str:
                 if atom.name in atoms:
                     lis.append(atom)
         model.delete(lis)

    def get_main_charges(self):
        """
        Gets the MM point charges for the system of interest

        Returns
        -------
        list 
            charges of system
    
        Examples
        --------
        >>> charges = get_main_charges()
        """

        return self.main_charges

    def convert_trajectory(self, traj):
        """
        Converts an OpenMM trajectory to get 
        topology and positions that are compatible with MDtraj
        
        Parameters
        ----------
        traj : OpenMM trajectory object

        Returns
        -------
        list
            positions in nm
        OpenMM topology object
                  
        Examples
        --------
        >>> positions, topology = convert_trajectory(OpenMM_traj)
        """

        topology = traj.topology.to_openmm()
        positions = traj.openmm_positions((0))
        
        return topology, positions


    def convert_input(self):
        """
        Converts inputs to OpenMM readable topologies and positions.
        Currently supports pdb inputs as well as Amber and Gromacs input files.
        """

        if self.system_info_format == 'pdb':
            # instantiate OpenMM pdb object
            if type(self.system_info) is list:
                self.pdb = OM_app.PDBFile(self.system_info[0])
            elif type(self.system_info) is str:
                self.pdb = OM_app.PDBFile(self.system_info)
            # instantiate OpenMM forcefield object
            self.forcefield = OM_app.ForceField(self.ff, self.ff_water)
            self.topology = self.pdb.topology
            self.positions = self.pdb.positions
            self.PeriodicBoxVector = self.topology.getPeriodicBoxVectors()

        elif self.system_info_format == 'Amber':
            for fil in self.system_info:
                if fil.endswith('prmtop'):
                    self.forcefield = OM_app.AmberPrmtopFile(fil)
                    self.topology = self.forcefield.topology
                    self.use_pdb = False
                if fil.endswith('pdb'):
                    self.pdb = OM_app.PDBFile(fil)
                    self.forcefield = OM_app.ForceField(self.ff, self.ff_water)
                    self.topology = self.pdb.topology
                    self.use_pdb = True
                if fil.endswith('inpcrd'):
                    self.inpcrd = OM_app.AmberInpcrdFile(fil)
                    self.positions = self.inpcrd.positions
                    self.PeriodicBoxVector = self.inpcrd.boxVectors

        elif self.system_info_format == 'Gromacs':
            for fil in self.system_info:
                if 'gro' in fil:
                    self.pdb = OM_app.GromacsGroFile(fil)
            for fil in self.system_info:
                if 'top' in fil:
                    self.forcefield = OM_app.GromacsTopFile(fil, periodicBoxVectors=self.pdb.getPeriodicBoxVectors())
                    self.topology = self.forcefield.topology


    def set_up_reporters(self, simulation):
        """
        Sets up reporters according to options 
        specified by arguments in md_param.
        See keywords in the Molecular Dynamics section of 
        the manual for more information.
        
        Parameters
        ----------
        simulation : OpenMM simulation object
            Adds reporters to simulation object
        """

        pot = False
        kin = False
        enrgy = False
        temp = False
        den = False

        if self.return_trajectory_interval != 0:
            if self.trajectory_format == 'NetCDF':
                simulation.reporters.append(NetCDFReporter(self.return_trajectory_filename, self.return_trajectory_interval))

        if self.return_checkpoint_interval != 0:
            simulation.reporters.append(OM_app.CheckpointReporter(self.return_checkpoint_filename, self.return_checkpoint_interval))
                
        if self.return_info:

            if 'potentialEnergy' in self.return_info:
                pot = True
            if 'kineticEnergy' in self.return_info:
                kin = True
            if 'totalEnergy' in self.return_info:
                enrgy = True
            if 'temperature' in self.return_info:
                temp = True
            if 'density' in self.return_info:
                den = True

            simulation.reporters.append(OM_app.StateDataReporter(self.return_info_filename, self.return_info_interval, step=True,
            potentialEnergy=pot, kineticEnergy=kin, totalEnergy=enrgy, temperature=temp, density=den))


    def set_external_charges(self):
        """
        Function not implemented for classes
        """
        pass
