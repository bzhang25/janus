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
    A wrapper class that calls OpenMM
    to obtain molecular mechanics information and 
    take steps in a molecular dynamics simulation. 
    Class inherits from MMWrapper.
    """

    def __init__(self, param):
        """
        Initializes an OpenMM wrapper class with a set of parameters for 
        running OpenMM. Creates an OpenMM pdb object using the given pdb_file
        and forcefield object using the given forcefield

        Parameters
        ----------
        param : dict 
            Parameters for molecular mechanics computations.
            Individual parameters include:
            - mm_pdb_file : a pdb file that contains the system of interest
            - mm_forcefield : the name of the forcefield to use, default is amber99sb.xml
            - mm_water_forcefield : the name of the forcefield to use for water, default is tip3p.xml
            - step_size : step size to integrate system in picoseconds, 
                            default is 0.002*OM_unit.picoseconds
            - integrator" : which integrator to use for simulation, default is Langevin
            - fric_coeff" : friction coefficient to couple the system to heat bath in inverse
                            picoseconds, default is 1/OM_unit.picosecond
            - temp: the temperature at which the simulation runs in kelvin, default is 300*OM_unit.kelvin
            - nonbondedMethod : method for nonbonded interactions, default is OM_app.NoCutoff
            - nonbondedCutoff : cutoff distance for nonbonded interactions in nanometers,
                                default is "1*OM_unit.nanometer",
            - constraints : which bonds and angles implemented with constraints,
                            default is OM_app.HBonds
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

            For more information about these pararmeters and 
            other possible parameter values consult docs.openmm.org

        """

        super().__init__(param, "OpenMM")

        self.ff = param['mm_forcefield']
        self.ff_water = param['mm_water_forcefield']
        self.nonbondMethod = eval(self.param['nonbondedMethod'])
        print(self.nonbondMethod)
        self.constraint = eval(self.param['constraints'])
        print(self.constraint)
        self.hMass = eval(self.param['hydrogenMass'])
        self.switchDis = eval(self.param['switchDistance']),

        self.NVE_integrator = param['NVE_integrator']
        self.NVT_integrator = param['NVT_integrator']

        if (type(param['md_steps']) is list and type(param['md_ensemble']) is list):
            self.md_ensemble = param['md_ensemble'][-1]
            self.other_md_ensembles = param['md_ensemble'][0:-1]
            self.other_ensemble_steps = param['md_steps'][0:-1]
        elif (type(param['md_steps']) is int and type(param['md_ensemble']) is str):
            self.md_ensemble = param['md_ensemble']
            self.other_md_ensembles = None
            self.other_ensemble_steps = None

        self.temp = param['temp']*OM_unit.kelvin
        self.step_size = param['step_size']*OM_unit.femtoseconds
        self.fric_coeff = param['fric_coeff']/OM_unit.picosecond

        if self.md_ensemble == 'NVT':
            self.integrator = self.NVT_integrator
        elif self.md_ensemble == 'NVE':
            self.integrator = self.NVE_integrator

        self.positions = None

        self.convert_input()

    def initialize(self, embedding_method):
        """
        Calls compute_info to get information for the system
        of interest in its initial state and saves the simulation 
        object and information dictionary returned by compute_info
    
        Parameters
        ----------
        embedding_method : str
            what embedding method to use for initialization.
            If 'Mechanical', all forces are included
            If 'Electrostatic', all coulomic forces are excluded

        """

        # should I minimize energy here? If so, need to return new positions

        if (self.other_md_ensembles is not None and self.other_ensemble_steps is not None):
            for i, ensemble in enumerate(self.other_md_ensembles):
                print('running equilibrating ensemble {} for {} steps'.format(ensemble,self.other_ensemble_steps[i]))
                
                if ensemble == 'NVT':
                    integrator = self.NVT_integrator
                elif ensemble == 'NVE':
                    integrator = self.NVE_integrator

                OM_system = self.create_openmm_system(self.topology)
                #OM_system = self.forcefield.createSystem(self.topology, nonbondedMethod=OM_app.CutoffPeriodic, nonbondedCutoff=0.8*OM_unit.nanometer, rigidWater=False)
                simulation, integrator_obj = self.create_openmm_simulation(OM_system, self.topology, self.pdb.positions, integrator, return_integrator=True)
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
            pos = self.pdb.positions

        print('starting main simulation')
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

        # Create an OpenMM simulation from the openmm system, topology, and positions.
        self.main_simulation = self.create_openmm_simulation(OM_system, self.topology, self.pdb.positions, self.integrator)

        with open(chkpt_file, 'rb') as f:
            self.main_simulation.context.loadCheckpoint(f.read())

        with open(restart_forces, 'rb') as force_file:
            force = pickle.load(force_file)

        
        self.set_up_reporters(self.main_simulation)
        # Calls openmm wrapper to get information specified
        self.main_info = OpenMMWrapper.get_state_info(self.main_simulation,
                                      energy=True,
                                      positions=True,
                                      forces=True)
        #print('before loading forces')
        #print(self.main_info)

        self.update_forces(force, self.qmmm_force, self.main_simulation)
        
        self.set_up_reporters(self.main_simulation)
        # Calls openmm wrapper to get information specified
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

        for f, coord in forces.items():
            coord *= MMWrapper.au_bohr_to_kjmol_nm             # convert this back to openmm units
            force_obj.setParticleParameters(f, f, coord)  # need to figure out if the first 2 parameters always the same or not

        force_obj.updateParametersInContext(simulation.context)  # update forces with qmmm force

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
        Gets the information for the system of interest

        Returns
        -------
        dict
            Information including the current energy, gradient, 
            and positions for the system of interest
    
        """
        
        return OpenMMWrapper.get_state_info(self.main_simulation, main_info=True)

    def compute_info(self, topology, positions, include_coulomb='all', initialize=False, return_system=False, return_simulation=False, link_atoms=None, minimize=False):
        """
        Gets information about a set of molecules as defined in the pdb, including energy, positions, forces

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
        system, simulation, state = compute_info(top, pos)
        state = compute_info(top, pos, return_simulation=False, return_system=False)
        """

        # ensure every computation has same periodic box vector parameters
        topology.setPeriodicBoxVectors(self.PeriodicBoxVector)
        # Create an OpenMM system from an object's topology
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
        forcefield, and other paramters as given in input
        TODO: need to put nonbond and nonbond_cutoff back but not doing for now
        because need non-periodic system. Other parameters are also needed
        also, expand forcefield to take not openmm built in
        but customized as well

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
        openmm_sys = create_openmm_system(topology)
        openmm_sys = create_openmm_system(top, include_coulomb='no_link', link_atoms=[0,1,2])
        openmm_sys = create_openmm_system(top, include_coulomb='only')
        openmm_sys = create_openmm_system(top, initialize=True)
        """

        # check to see if there are unmatched residues in pdb, create residue templates if there are
        unmatched = self.forcefield.getUnmatchedResidues(topology)
        if unmatched:
            self.create_new_residue_template(topology)

        openmm_system = self.forcefield.createSystem(topology,
                                        nonbondedMethod=self.nonbondMethod,
                                        constraints=self.constraint,
                                        hydrogenMass=self.hMass,
                                        switchDistance=self.switchDis,
                                        residueTemplates=self.param['residueTemplates'],
                                        nonbondedCutoff=self.param['nonbondedCutoff']*OM_unit.nanometer,
                                        rigidWater=self.param['rigid_water'],
                                        removeCMMotion=self.param['removeCMMotion'],
                                        flexibleConstraints=self.param['flexibleConstraints'],
                                        ignoreExternalBonds=self.param['ignoreExternalBonds'])


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
        set_charge_zero(system)
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
        create_new_residue_template(topology)
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
            print(name)
            if name in self.forcefield._templates:
                print('override existing modified residues with same name')
                template[i].overrideLevel = self.forcefield._templates[name].overrideLevel + 1

            # register the new template to the forcefield object
            print('register the new template to the forcefield object')
            self.forcefield.registerResidueTemplate(template[i])


    def create_openmm_simulation(self, openmm_system, topology, positions, integrator,  return_integrator=False):
        """
        Creates an OpenMM simulation object given
        an OpenMM system, topology, and positions

        Parameters
        ----------
        openmm_system : OpenMM system object
        topology : OpenMM topology object
        positions : OpenMM Vec3 vector 
            contains the positions of the system in nm

        Returns
        -------
        OpenMM simulation object

        Examples
        --------
        create_open_simulation(openmm_sys, top, pos) 
        create_open_simulation(openmm_sys, pdb.topology. pdb.positions)
        """

        print('using {} integrator'.format(integrator))
        if integrator == 'Langevin':
            integrator_obj = OM.LangevinIntegrator(self.temp, self.fric_coeff, self.step_size)
            integrator_obj.setRandomNumberSeed(1)
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
        get_state_info(sim)
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
        Write a pdb file from an OpenMM modeller

        """
        return_sys  = False
        sys_file    = 'final.pdb'

        try:
            return_sys = self.param['return_system']
        except:
            print("Will not return final pdb")

        if return_sys is True: 
            try:
                sys_file = self.param['return_system_filename']
            except:
                print('writing final pdb to final.pdb')
                
            OM_app.PDBFile.writeFile(info['topology'], info['positions'], open(sys_file, 'w'))
 

    def create_modeller(self, qm_atoms, keep_qm=None):
        """
        Makes a OpenMM modeller object based on given geometry

        Parameters
        ----------
        keep_qm : bool 
            whether to keep the qm atoms in the modeller or delete them.
            The default is to make a modeller without the qm atoms

        Returns
        -------
        A OpenMM modeller object

        Examples
        --------
        modeller = self.make_modeller()
        modeller = self.make_modeller(keep_qm=True)
        """

        modeller = OM_app.Modeller(self.topology, self.pdb.getPositions())
        if keep_qm is False:
            OpenMMWrapper.delete_atoms(modeller, qm_atoms)
        elif keep_qm is True:
            OpenMMWrapper.keep_atoms(modeller, qm_atoms)
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
            keep_atom(mod, [0,1])
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
         delete_atoms(model, [0, 3, 5])
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
        charges = get_main_charges()
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
        positions, topology = convert_trajectory(OpenMM_traj)
        """

        topology = traj.topology.to_openmm()
        positions = traj.openmm_positions((0))
        
        return topology, positions


    def set_external_charges(self):
        pass

    def convert_input(self):

        if self.system_info_format == 'pdb':
            # instantiate OpenMM pdb object
            self.pdb = OM_app.PDBFile(self.system_info[0])
            # instantiate OpenMM forcefield object
            self.forcefield = OM_app.ForceField(self.ff, self.ff_water)
            self.topology = self.pdb.topology
            
            self.PeriodicBoxVector = self.topology.getPeriodicBoxVectors()

        elif self.system_info_format == 'Amber':
            for fil in self.system_info:
                if 'prmtop' in fil:
                    self.forcefield = OM_app.AmberPrmtopFile(fil)
                    self.topology = self.forcefield.topology
                elif 'inpcrd' in fil:
                    self.pdb = OM_app.AmberInpcrdFile(fil)
                    self.boxVectors = self.pdb.boxVectors

        elif self.system_info_format == 'Gromacs':
            for fil in self.system_info:
                if 'gro' in fil:
                    self.pdb = OM_app.GromacsGroFile(fil)
            for fil in self.system_info:
                if 'top' in fil:
                    self.forcefield = OM_app.GromacsTopFile(fil, periodicBoxVectors=self.pdb.getPeriodicBoxVectors())
                    self.topology = self.forcefield.topology


    def set_up_reporters(self, simulation):

        pot = False
        kin = False
        enrgy = False
        temp = False
        den = False

        if self.param['return_checkpoint_interval']:
            return_chkpt_int = self.param['return_checkpoint_interval']
        else:
            return_chkpt_int = 0
        if self.param['return_checkpoint_filename']:
            chkpt_file = self.param['return_checkpoint_filename']
        else:
            chkpt_file = 'checkpoint.chk'
        if self.param['return_trajectory_interval']:
            return_traj_int = self.param['return_trajectory_interval']
        else:
            return_traj_int = 0
        if self.param['return_trajectory_filename']:
            traj_file = self.param['return_trajectory_filename']
        else:
            traj_file   = 'output.nc'
        if self.param['trajectory_format']:
            traj_format = self.param['trajectory_format']
        else:
            traj_format = 'NetCDF'
        if self.param['return_info_interval']:
            info_int = self.param['return_info_interval']
        else:
            info_int    = 0
        if self.param['return_info']:
            return_info = self.param['return_info']
        else:
            return_info = []

        if return_traj_int != 0:
            if traj_format == 'NetCDF':
                simulation.reporters.append(NetCDFReporter(traj_file, return_traj_int))

        if return_chkpt_int != 0:
            simulation.reporters.append(OM_app.CheckpointReporter(chkpt_file, return_chkpt_int))

        if return_info:

            if 'potentialEnergy' in return_info:
                pot = True
            if 'kineticEnergy' in return_info:
                kin = True
            if 'totalEnergy' in return_info:
                enrgy = True
            if 'temperature' in return_info:
                temp = True
            if 'density' in return_info:
                den = True
                
            simulation.reporters.append(OM_app.StateDataReporter('info.dat', info_int, step=True,
            potentialEnergy=pot, kineticEnergy=kin, totalEnergy=enrgy, temperature=temp, density=den))



