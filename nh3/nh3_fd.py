import os
import numpy
from pyscf import gto, scf, cc, fci
import QMCUtils
import numpy as np
from functools import partial
import h5py
from ad_afqmc import pyscf_interface, driver

print = partial(print, flush=True)

r = 1.012
theta = 106.67 * np.pi / 180.
rz = r * np.sqrt(np.cos(theta/2)**2 - np.sin(theta/2)**2/3)
dc = 2 * r * np.sin(theta/2) / np.sqrt(3)
atomstring = f'''
                 N 0. 0. 0.
                 H 0. {dc} {rz}
                 H {r * np.sin(theta/2)} {-dc/2} {rz}
                 H {-r * np.sin(theta/2)} {-dc/2} {rz}
              '''
mol = gto.M(atom=atomstring, basis='ccpvdz', verbose=3, symmetry=0)
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 1
# dipole moment
nuc_dipmom = [0.0, 0.0, 0.0]
for i in range(mol.natm):
  for j in range(3):
    nuc_dipmom[j] += mol.atom_charge(i) * mol.atom_coord(i)[j]

dip_ints_ao = -mol.intor_symmetric('int1e_r', comp=3)
dip_ints_mo = np.empty_like(dip_ints_ao)
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo[i] = mf.mo_coeff.T.dot(dip_ints_ao[i]).dot(mf.mo_coeff)

# fd
e_afqmc = [0., 0., 0., 0., 0., 0.]
e_err_afqmc = [0., 0., 0., 0., 0., 0.]
e_rhf = [0., 0., 0., 0., 0., 0.]
dE = 0.05

options = {'n_eql': 2,
           'n_ene_blocks': 1,
           'n_sr_blocks': 5,
           'n_blocks': 100,
           'n_walkers': 50,
           'seed': 15902}

for i, pm in enumerate([-3, -2, -1, 1, 2, 3]):
  mf = scf.RHF(mol)
  h1e = mf.get_hcore()
  h1e += pm * dE * dip_ints_ao[2]
  mf.get_hcore = lambda *args: h1e
  mf.kernel()
  e_rhf[i] = mf.e_tot

  pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen)
  e_afqmc[i], e_err_afqmc[i] = driver.run_afqmc(options=options)
  e_afqmc[i] += pm * dE * nuc_dipmom[2]
  print(f"Finished AFQMC / RHF calculation {i}\n")

print(f'\ne_rhf: {e_rhf}')
print(f'e_afqmc: {e_afqmc}')
print(f'e_err_afqmc: {e_err_afqmc}\n')

obs_afqmc_rhf_3p = (-e_afqmc[2] + e_afqmc[3]) / 2. / dE
obs_afqmc_rhf_3p_err = (e_err_afqmc[2]**2 + e_err_afqmc[3]**2)**0.5 / 2. / dE
obs_afqmc_rhf_5p = (e_afqmc[1] - 8 * (e_afqmc[2] - e_afqmc[3]) - e_afqmc[4]) / 12. / dE
obs_afqmc_rhf_5p_err = (e_err_afqmc[1]**2 + 64 * (e_err_afqmc[2]**2 + e_err_afqmc[3]**2) + e_err_afqmc[4]**2)**0.5 / 12. / dE
obs_afqmc_rhf_7p = (-e_afqmc[0] + 9 * (e_afqmc[1] - e_afqmc[4]) - 45 * (e_afqmc[2] - e_afqmc[3]) + e_afqmc[5]) / 60. / dE
obs_afqmc_rhf_7p_err = (e_err_afqmc[0]**2 + 9**2 * (e_err_afqmc[1]**2 + e_err_afqmc[4]**2) + 45**2 * (e_err_afqmc[2]**2 + e_err_afqmc[3]**2) + e_err_afqmc[5]**2)**0.5 / 60. / dE

print(f'obs_rhf_3p = {(-e_rhf[2] + e_rhf[3]) / 2. / dE}')
print(f'obs_rhf_5p = {(e_rhf[1] - 8 * (e_rhf[2] - e_rhf[3]) - e_rhf[4]) / 12. / dE}')
print(f'obs_rhf_7p = {(-e_rhf[0] + 9 * (e_rhf[1] - e_rhf[4]) - 45 * (e_rhf[2] - e_rhf[3]) + e_rhf[5]) / 60. / dE}')
print(f'obs_afqmc_rhf_3p = {obs_afqmc_rhf_3p} +/- {obs_afqmc_rhf_3p_err}')
print(f'obs_afqmc_rhf_5p = {obs_afqmc_rhf_5p} +/- {obs_afqmc_rhf_5p_err}')
print(f'obs_afqmc_rhf_7p = {obs_afqmc_rhf_7p} +/- {obs_afqmc_rhf_7p_err}')

