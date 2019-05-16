import json
import os
from janus.qm_wrapper import Psi4Wrapper 
from janus.mm_wrapper import OpenMMWrapper 
from janus.qmmm import QMMM, OniomXS, HotSpot, PAP, SAP, DAS
import time

class Initializer(object):
    """
    Class that initializes the initial parameters 
    and necessary wrapppers.

    Parameters
    ----------
    paramaters : str 
        filename of a json input file; if as_file is False, param is a dict
    as_file : bool
            param is a file if True, dict if False

    """

    def __init__(self, parameters, as_file=True):

        self.system = {}
        self.qmmm = {}
        self.aqmmm = {}
        self.md = {} 
        self.ll = {} 
        self.hl = {} 

        self.system_info_format = 'pdb'
        self.run_qmmm = True
        self.run_aqmmm = False
        self.run_md = False
        self.aqmmm_scheme = None
        self.md_restart = False
        self.ll_program = 'OpenMM'
        self.hl_program = 'Psi4'
        self.md_simulation_program = "OpenMM"
        self.md_restart_checkpoint_filename = 'checkpoint.chk'
        self.md_restart_forces_filename = 'forces.pkl'

        if as_file is True:
            self.param = self.load_param(parameters)
        else:
            self.param = parameters

        self.set_attributes(self.param, self)
        self.aqmmm['qmmm_param'] = self.qmmm

        if self.system is not None and 'system_info' in self.system:
            self.set_attributes(self.system, self)
        else:
            raise KeyError("system_info needs to be specified")

        if self.md:
            self.ll.update(self.md)

        self.get_wrappers()
    
    def get_wrappers(self):
        """
        Determines what type of wrapper object to instantiate 
        based on input parameters

        """

        if self.hl_program == "Psi4":
            self.hl_wrapper = Psi4Wrapper
        elif self.hl_program == "OpenMM":
            self.hl_wrapper = OpenMMWrapper
        else:
            raise ValueError("Only Psi4 and OpenMM currently available to be used in high level computations")

        if self.ll_program == "OpenMM":
            self.ll_wrapper = OpenMMWrapper
        else:
            raise ValueError("Only OpenMM currently available to be used in low level computations")

        if self.run_md is True:

            if self.md_simulation_program == "OpenMM":
                self.md_sim_wrapper = OpenMMWrapper
            else:
                raise ValueError("Only OpenMM currently available")

        #self.md_sim_param.update(self.system_param)


    def initialize_wrappers(self):
        """
        Instantiates wrapper objects based on input parameters.
        Create instances of appropriate wrappers for 
        low and high level computations, QM/MM or adaptive QM/MM computations,
        as well as MD simulations.

        Returns
        -------
        :class:`~janus.mm_wrapper.MMWrapper` subclass
        :class:`~janus.qmmm.QMMM` or :class:`~janus.qmmm.AQMMM` subclass

        """

        # create hl_wrapper object
        hl_wrapper = self.hl_wrapper(sys_info=self.system_info, sys_info_format=self.system_info_format, **self.hl)
        # create ll_wrapper object
        ll_wrapper = self.ll_wrapper(sys_info=self.system_info, sys_info_format=self.system_info_format, **self.ll)
        
        if (self.run_aqmmm is True and self.aqmmm_scheme is None):
            raise Exception("AQMMM specified but no scheme was given")
        if self.aqmmm_scheme is None:
            qmmm_wrapper = QMMM(hl_wrapper, ll_wrapper, self.system_info, sys_info_format=self.system_info_format, **self.qmmm)
        elif self.aqmmm_scheme == 'ONIOM-XS':
            qmmm_wrapper = OniomXS(hl_wrapper, ll_wrapper, self.system_info, self.system_info_format, **self.aqmmm)
        elif self.aqmmm_scheme == 'Hot-Spot':
            qmmm_wrapper = HotSpot(hl_wrapper, ll_wrapper, self.system_info, self.system_info_format, **self.aqmmm)
        elif self.aqmmm_scheme == 'PAP':
            qmmm_wrapper = PAP(hl_wrapper=hl_wrapper, ll_wrapper=ll_wrapper, sys_info=self.system_info, sys_info_format=self.system_info_format, **self.aqmmm)
        elif self.aqmmm_scheme == 'SAP':
            qmmm_wrapper = SAP(hl_wrapper=hl_wrapper, ll_wrapper=ll_wrapper, sys_info=self.system_info, sys_info_format=self.system_info_format, **self.aqmmm)
        elif self.aqmmm_scheme == 'DAS':
            qmmm_wrapper = DAS(hl_wrapper=hl_wrapper, ll_wrapper=ll_wrapper, sys_info=self.system_info, sys_info_format=self.system_info_format, **self.aqmmm)
        else:
            raise ValueError("{} not recognized as a currently implemented method".format(self.aqmmm_param['aqmmm_scheme']))

        if self.run_md is True:

            md_sim_wrapper = self.md_sim_wrapper(sys_info=self.system_info, sys_info_format=self.system_info_format, **self.ll)

            
            t1 = time.time()
            if self.md_restart is False:
                # initialize mm_wrapper with information about initial system
                md_sim_wrapper.initialize(qmmm_wrapper.embedding_method)
            else:
                md_sim_wrapper.restart(qmmm_wrapper.embedding_method, 
                                       self.md_restart_checkpoint_filename,
                                       self.md_restart_forces_filename)
            t2 = time.time()
            time_all = t2 - t1 + qmmm_wrapper.time_zero_energy

            return md_sim_wrapper, qmmm_wrapper, time_all
 
        else:
            return ll_wrapper, qmmm_wrapper, qmmm_wrapper.time_zero_energy
        

    def load_param(self, fname):
        """
        Converts a json file into a dictionary
    
        Parameters
        ----------
        fname : str
            name of json file

        Returns
        -------
        dict
            parameters contained in filename

        """

        if fname.endswith(".json"):
            rfunc = json.load
        elif fname.endswith(".yaml"):
            rfunc = yaml.load
        elif fname.endswith(".yml"):
            rfunc = yaml.load
        else:
            raise TypeError("Did not understand file type {}.".format(fname))

        with open(fname, 'r') as handle:
            ret = rfunc(handle)

        return ret


    def set_attributes(self, dictionary, obj):
        """
        Sets values in the dicionary to be 
        an attribute of the object given.
        The key of the dictionary will be set as 
        an attribute of the object given.

        Parameters
        ----------
        dictionary : dict
            Dictionary with relevant parameters
        obj : obj
            Object to have attributes set to.

        Returns
        -------
        None

        Examples
        --------    
        >>> set_attributes({'aqmmm' : False}, self)
        
        This sets self.aqmmm to be False.

        """

        for k, v in dictionary.items():
            setattr(obj, k, v)


