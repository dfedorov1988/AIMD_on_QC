import numpy as np
import matplotlib.pyplot as plt
from h2_md import compute_energy_and_grad
import h5py
import os.path
from os import path
import sys
import time
from qiskit.algorithms.optimizers import L_BFGS_B, SLSQP
import logging

def prop_first_step(x_t, v_t, t, dt, m, dx, shots, parameters=None, map_type="parity", qubit_reduction=False,
                    backend='statevector_simulator', optimizer="SLSQP"):

    pote_t, e_p_dx_t, e_m_dx_t, e_fci_t, grad_t, grad_fci_t, parameters,\
            = compute_energy_and_grad(x_t, dx,
        shots, qubit_reduction=qubit_reduction, backend_type=backend,
        map_type=map_type,  parameters=parameters, opt=optimizer)
    a_t = -grad_t / m

    ke_t = m * v_t ** 2 / 2

    print("\nt =", t)
    # initial output
    times.append(0)
    pos.append(x_t)
    kinen.append(ke_t)
    fci_en.append(e_fci_t)
    poten.append(pote_t)
    forces.append(-grad_t)
    toten.append(ke_t + pote_t)
    velocities.append(v_t)

    v_tphdt = v_t + 0.5 * a_t * dt
    x_tpdt = x_t + v_tphdt * dt
    print("x_tpdt", x_tpdt)


    pote_tpdt, e_p_dx_tpdt, e_m_dx_tpdt, e_fci_tpdt,  grad_tpdt, grad_fci_tpdt,  parameters\
         = compute_energy_and_grad(x_tpdt, dx,
        shots,  qubit_reduction=qubit_reduction, backend_type=backend,
        map_type=map_type, parameters=parameters, opt=optimizer)

    a_tpdt = -grad_tpdt / m

    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    ke_tpdt = m * v_tpdt ** 2 / 2
    t += dt
    print("t =", t)

    # output
    times.append(t)
    pos.append(x_tpdt)
    fci_en.append(e_fci_tpdt)
    kinen.append(ke_tpdt)
    poten.append(pote_tpdt)
    forces.append(-grad_tpdt)
    toten.append(ke_tpdt + pote_tpdt)
    velocities.append(v_tpdt)
    params.append(parameters)

    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

    print("x_tpdt", x_tpdt)

    return x_tpdt, a_tpdt, v_tp3hdt, t, parameters


def prop_not_first_step(x_t, v_tphdt, t, dt, m, dx, shots, parameters=None,  map_type="parity",
                        qubit_reduction=False, backend="statevector_simulator", optimizer='COBYLA'):

    x_tpdt = x_t + v_tphdt * dt

    pote_tpdt, e_p_dx_tpdt, e_m_dx_tpdt, e_fci_tpdt,  grad_tpdt, grad_fci_tpdt, parameters,\
            = compute_energy_and_grad(x_tpdt, dx,
        shots, qubit_reduction=qubit_reduction,
        map_type=map_type,  parameters=parameters, backend_type=backend, opt=optimizer)
    
    a_tpdt = -grad_tpdt / m

    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    ke_tpdt = m * v_tpdt ** 2 / 2
    t += dt
    print("\nt =", t)

    times.append(t)
    pos.append(x_tpdt)
    kinen.append(ke_tpdt)
    fci_en.append(e_fci_tpdt)
    poten.append(pote_tpdt)
    forces.append(-grad_tpdt)
    velocities.append(v_tpdt)
    toten.append(ke_tpdt + pote_tpdt)

    params.append(parameters)
    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

    return x_tpdt, v_tp3hdt, t, parameters

def append_to_hdf5(hdf5_filename, times, pos, kinen, fci_en, poten, forces, velocities, toten, parameters):
    
    if path.exists(hdf5_filename):
        os.remove(hdf5_filename)
    with h5py.File(hdf5_filename, "w") as hf:
        hf.create_dataset('times', data=times)
        hf.create_dataset('poten', data=poten)
        hf.create_dataset('kinen', data=kinen)
        hf.create_dataset('fci_en', data=fci_en)
        hf.create_dataset('toten', data=toten)
        hf.create_dataset('pos', data=pos)
        hf.create_dataset('forces', data=forces)
        hf.create_dataset('velocities', data=velocities)
        hf.create_dataset('optimized_parameters', data=params)

        hf.attrs['x_ini'] = x_ini
        hf.attrs['v_ini'] = v_ini
        hf.attrs['dt'] = dt
        hf.attrs['dx'] = dx
        hf.attrs['t_ini'] = t_ini
        hf.attrs['map_type'] = map_type
        hf.attrs['qubit_reduction'] = qubit_reduction
        hf.attrs['shots'] = shots
    
def read_hdf5(hdf5_filename):
    hf = h5py.File(hdf5_filename, 'r')

    for attr in hf.attrs:
        print(attr, hf.attrs[attr])

    times = np.array(hf.get('times'))
    poten = np.array(hf.get('poten'))
    toten = np.array(hf.get('toten'))
    fcien = np.array(hf.get('fci_en'))
    kinen = np.array(hf.get('kinen'))
    forces = np.array(hf.get('forces'))
    pos = np.array(hf.get('pos'))


hdf5_filename = sys.argv[1]

tic = time.perf_counter()

x_ini = 2.0  # initial position
v_ini = 0.0  # initial velocity
m = 1822.0  # mass of H in atomic units
dt = 5.0  # timestep in au
dx = 1e-05  # displacement for numerical gradient
t_ini = 0  # starting time
sim_time = 300 # total sim time in au

qubit_reduction = False
map_type = "parity"
shots = 8192
backend = 'statevector_simulator'
restart = False
optimizer = SLSQP()
ini_params = [ 0.01373469, -0.02186433, -0.30696973]

if path.exists(hdf5_filename):
    if not restart:
        print(hdf5_filename, "file exists in current directory, rename or move it to avoid losing data")
        sys.exit()
    if restart:
        x_tpdt, v_tp_3hdt, t, dt, dx, shots, parameters, qubit_reduction, map_type, backend =\
            read_hdf5(hdf5_filename)

# arrays to store trajectory info
times = []
pos = []
fci_en = []
poten = []
kinen = []
toten = []
forces = []
velocities = []
params = []

# Preparations for the first time step
x_t = x_ini
t = t_ini
v_t = v_ini
x_tpdt, a_tpdt, v_tp3hdt, t, parameters = prop_first_step(x_t, v_t, t, dt, m, dx, shots,
                                                          qubit_reduction=qubit_reduction, map_type=map_type,
                                                          backend=backend, parameters=ini_params,
                                                          optimizer=optimizer)

append_to_hdf5(hdf5_filename, times, pos, kinen, fci_en, poten, forces, velocities, toten, parameters)

# Propagating until the desired time using not_first_step
while t < sim_time:
    x_tpdt, v_tp3hdt, t, parameters = prop_not_first_step(x_tpdt, v_tp3hdt, t, dt, m, dx, shots, parameters=parameters,
                                                          qubit_reduction=qubit_reduction,
                                                          map_type=map_type, backend=backend, 
                                                          optimizer=optimizer)
    append_to_hdf5(hdf5_filename, times, pos, kinen, fci_en, poten, forces, velocities, toten, parameters)

# Clocking the total wall time
toc = time.perf_counter()
total_wall_time = toc - tic
print("Total time in seconds:", total_wall_time)

print("Simulation is done! Data dumped into " + hdf5_filename +  " file")
plt.plot(times, toten)
plt.show()