"""
This is the qmmm driver module
"""
import pickle
from janus import Initializer
import time
    
def run_janus(filename='input.json'):
    """
    Drives the janus program.
    Creates an instance of the Initializer class 
    and feeds wrappers to either :func:`~janus.driver.run_simulation` or
    :func:`~janus.driver.run_single_point` 
    
    Parameters
    ----------

    filename : str
        Filename from which to read input parameters

    """

    t0 = time.time()
    initializer = Initializer(filename)

    print('Initializing')
    # initialize wrappers
    ll_wrapper, qmmm_wrapper, t_int = initializer.initialize_wrappers()

    if initializer.run_md is True:
        t_ext = run_simulation(ll_wrapper, qmmm_wrapper)
    else:
        t_ext = run_single_point(ll_wrapper, qmmm_wrapper)

    t1 = time.time()
    print('Simulation took {:10.5f} seconds'.format(t1-t0))
    print('Time spent in external programs: {:10.5f} seconds'.format(t_ext + t_int))
    print('Overhead time: {:10.5f} seconds'.format(t1 - t0 - t_ext - t_int))

def run_simulation(md_sim_wrapper, qmmm_wrapper):
    """
    Drives QM/MM with MD time step integration
    
    Parameters
    ----------
    md_sim_wrapper : :class:`~janus.mm_wrapper.MMWrapper`
        A child class of MMWrapper that drives MD simulation
    qmmm_wrapper: :class:`~janus.qmmm.QMMM`
        A QMMM or AQMMM wrapper that drives the QM/MM computations
    """
    t_md = 0
    t_qmmm = 0

    t_md1 = time.time() 
    print('Equilibrating with {} steps'.format(md_sim_wrapper.start_qmmm))
    md_sim_wrapper.take_step(md_sim_wrapper.start_qmmm)

    t_md2 = time.time() 
    t_md += t_md2 - t_md1

    for step in range(md_sim_wrapper.qmmm_steps):

        print('Taking step {}'.format(step + 1))
        t = run_single_point(md_sim_wrapper, qmmm_wrapper)
        t_qmmm += t
        
        # get aqmmm forces 
        forces = qmmm_wrapper.get_forces()

        t_md3 = time.time() 

        if (md_sim_wrapper.return_forces_interval != 0 and (step + 1) % md_sim_wrapper.return_forces_interval == 0):
            with open(md_sim_wrapper.return_forces_filename, 'wb') as f:
                pickle.dump(forces, f)

        # feed forces into md simulation and take a step
        # make sure positions are updated so that when i get information on entire system 
        # getting it on the correct one
        md_sim_wrapper.take_updated_step(force=forces)

        t_md4 = time.time() 
        t_md += t_md4 - t_md3

    print('QMMM finished')

    t_md5 = time.time() 

    md_sim_wrapper.take_step(md_sim_wrapper.end_steps)
    main_info = md_sim_wrapper.get_main_info()
    md_sim_wrapper.write_pdb(main_info)

    t_md6 = time.time() 
    t_md += t_md6 - t_md5

    return t_md + t_qmmm


def run_single_point(ll_wrapper, qmmm_wrapper):
    """
    Drives single QM/MM computation

    Parameters
    ----------
    ll_wrapper : :class:`~janus.mm_wrapper.MMWrapper`
        A child class of MMWrapper that contains MM information on the whole system
    qmmm_wrapper: :class:`~janus.qmmm.QMMM`
        A QMMM or AQMMM wrapper that drives the QM/MM computations

    """
    #get MM information for entire system
    t_md0 = time.time()
    main_info = ll_wrapper.get_main_info()
    t_md1 = time.time()

    t_qmmm = qmmm_wrapper.run_qmmm(main_info, ll_wrapper.class_type)

    return t_md1 - t_md0 + t_qmmm

