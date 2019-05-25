import psi4
import numpy as np
from janus.qm_wrapper import QMWrapper

class Psi4Wrapper(QMWrapper):
    """
    A wrapper class that calls Psi4 to obtain quantum mechanical
    information. Class inherits from QMWrapper.
    """

    def __init__(self,
                 method='scf',
                 charge=0,
                 multiplicity=1,
                 reference='rhf',
                 basis='STO-3G',
                 e_convergence=1e-8,
                 d_convergence=1e-8,
                 sys_info=None,
                 sys_info_format=None,
                 **kwargs):
        """
        Initializes a Psi4Wrapper class with a set of 
        parameters for running Psi4

        Parameters
        ----------
        param : dict 
            parameters for QM computations
            Individual parameters include:

            - basis_set : the basis set to use for compuatations, 
                        default is STO-3G
            - scf_type : scf algorithm, default is density fitting(df)
            - guess_orbitals : type of guess orbitals, default is 
                                Superposition of Atomic Densities(sad)
            - reference : type of reference wavefunction, default is RHF
            - e_convergence : degree of energy convergence, default is 1e-8
            - d_convergence : degree of density convergence, default is 1e-8
            - method : computation method, default is scf
            - charge_method : method for getting QM charges, default is Mulliken
            - charge : charge of qm system, default is 0
            - multiplicity : spin state of qm system, default is singlet(1)

            For more information about these parameters and 
            other possible parameter values consult psicode.org

        """

        super().__init__("Psi4")
        self.energy = None
        self.wavefunction = None
        self.gradient = None

        self.method = method
        self.charge = charge
        self.multiplicity = multiplicity

        self.qm_param = kwargs
        self.qm_param['reference'] = reference
        self.qm_param['basis'] = basis
        self.qm_param['e_convergence'] = e_convergence
        self.qm_param['d_convergence'] = d_convergence

    def compute_energy(self):
        """
        Calls Psi4 to obtain the energy and Psi4 wavefunction object of the QM region
        and saves as self.energy and self.wavefunction
        """
        self.set_up_psi4()
        self.energy, self.wavefunction = psi4.energy(self.method,
                                                       return_wfn=True)

    def compute_gradient(self):
        """
        Calls Psi4 to obtain the gradient of the QM region
        and saves it as a numpy array self.gradient
        """
        self.set_up_psi4()
        G = psi4.gradient(self.method)
        self.gradient = np.asarray(G)

    def compute_info(self):
        """
        Calls Psi4 to obtain the energy, Psi4 wavefunction object, and 
        gradient of the QM region and saves as self.energy, self.wavefuction,
        and self.gradient
        """
        try:
            self.set_up_psi4()
            self.energy, self.wavefunction = psi4.energy(self.method,
                                                        return_wfn=True)

            G = psi4.gradient(self.method)
            self.gradient = np.asarray(G)
        except:
            print('in exception block')
            #print('scf failed to converge most likely, trying something else')
            
            #self.set_up_psi4(be_quiet=False)
            #self.energy, self.wavefunction = psi4.energy(self.method,
            #                                            return_wfn=True)

            #G = psi4.gradient(self.method)
            #self.gradient = np.asarray(G)
            
        #deriv = psi4.core.Deriv(self.wavefunction)
        #deriv.compute()
        #self.gradient = np.asarray(self.wavefunction.gradient())

    def optimize_geometry(self):
        """
        Calls Psi4 to obtain a geometry optimized geometry 
    
        Returns
        -------
        numpy array
            XYZ coordinates of the optimized geometry
        """

        self.set_up_psi4()
        self.energy, self.wavefunction = psi4.opt(self.method, return_wfn=True)
        return np.array(self.wavefunction.molecule().geometry())

    def set_up_psi4(self, be_quiet=True):
        """
        Sets up a psi4 computation
        """
        # psi4.core.set_output_file('output.dat', True)
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.EXTERN = None 
        
        # Supress print out
        if be_quiet is True:
            psi4.core.be_quiet()
        
        psi4.set_options(self.qm_param)

        psi4_geom = '\n' + str(self.charge) + ' ' + str(self.multiplicity) + '\n '
        psi4_geom += self.qm_geometry
        psi4_geom += 'no_reorient \n'
        psi4_geom += 'no_com \n '
        #print(psi4_geom)
        print('psi4 geometry')
        print(psi4_geom)

        # make sure this is in angstroms
        mol = psi4.geometry(psi4_geom)

        if self.external_charges is not None:
            Chrgfield = psi4.QMMM()
            for charge in self.external_charges:
                Chrgfield.extern.addCharge(charge[0], charge[1], charge[2], charge[3])
            psi4.core.set_global_option_python('EXTERN', Chrgfield.extern)

            
    def compute_scf_charges(self, charge_method='MULLIKEN_CHARGES'):
        """
        Calls Psi4 to obtain the self.charges on each atom given and saves it as a numpy array.
        This method works well for SCF wavefunctions. For correlated levels of theory (e.g., MP2),
        it is advised that compute_energy_and_charges() be used instead.
        """
        if self.wavefunction is not None:
            psi4.oeprop(self.wavefunction, charge_method)
            self.charges = np.asarray(self.wavefunction.atomic_point_charges())
            self.charges = self.charges 


    def compute_energy_and_charges(self, charge_method='MULLIKEN_CHARGES'):
        """
        Calls Psi4 to obtain the self.energy, self.wavefunction, 
        and self.charges on each atom. This method for correlated methods.
        
        Note
        ----
        Think about passing in wavefunction instead of calling for energy and wavefunction
        """
        self.set_up_psi4()
        self.energy, self.wavefunction = psi4.prop(self.method,
                                properties=[charge_method],
                                return_wfn=True)
        self.charges = np.asarray(self.wavefunction.atomic_point_charges())


    def build_qm_param(self):
        """
        Builds a dictionary of QM parameters from input options
        and saves as self.param
        """

        #if self.is_open_shelled is True and self.reference == 'rhf':
        #    qm_param['reference'] = 'uhf'
        #    self.multiplicity = 2
        #else:
        #    qm_param['reference'] = self.reference
        return self.qm_param



