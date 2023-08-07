import os
import numpy
from pyscf import gto, scf, cc, fci
import QMCUtils
import numpy as np
from functools import partial
import h5py
from ad_afqmc import pyscf_interface, driver

print = partial(print, flush=True)

r = 2.4
nH = 20
atomstring = ""
for i in range(nH):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(atom=atomstring, basis='sto-6g', verbose=3, unit='bohr')
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 0

# fd
e_afqmc = [0., 0., 0., 0., 0., 0.]
e_err_afqmc = [0., 0., 0., 0., 0., 0.]
e_hf = [0., 0., 0., 0., 0., 0.]
dE = 0.05

options = {'n_eql': 2,
           'n_ene_blocks': 1,
           'n_sr_blocks': 5,
           'n_blocks': 100,
           'n_walkers': 50,
           'walker_type': 'uhf',
           'seed': 9802}

for i, pm in enumerate([-3, -2, -1, 1, 2, 3]):
  umf = scf.UHF(mol)
  h1e = umf.get_hcore()
  h1e += pm * dE * h1e
  umf.get_hcore = lambda *args: h1e
  umf.kernel()
  mo1 = umf.stability(external=True)[0]
  umf = umf.newton().run(mo1, umf.mo_occ)
  mo1 = umf.stability(external=True)[0]
  umf = umf.newton().run(mo1, umf.mo_occ)
  e_hf[i] = umf.e_tot

  pyscf_interface.prep_afqmc(umf, norb_frozen=norb_frozen)
  e_afqmc[i], e_err_afqmc[i] = driver.run_afqmc(options=options)
  print(f"Finished AFQMC / hf calculation {i}\n")

print(f'\ne_hf: {e_hf}')
print(f'e_afqmc: {e_afqmc}')
print(f'e_err_afqmc: {e_err_afqmc}\n')

obs_afqmc_hf_3p = (-e_afqmc[2] + e_afqmc[3]) / 2. / dE
obs_afqmc_hf_3p_err = (e_err_afqmc[2]**2 + e_err_afqmc[3]**2)**0.5 / 2. / dE
obs_afqmc_hf_5p = (e_afqmc[1] - 8 * (e_afqmc[2] - e_afqmc[3]) - e_afqmc[4]) / 12. / dE
obs_afqmc_hf_5p_err = (e_err_afqmc[1]**2 + 64 * (e_err_afqmc[2]**2 + e_err_afqmc[3]**2) + e_err_afqmc[4]**2)**0.5 / 12. / dE
obs_afqmc_hf_7p = (-e_afqmc[0] + 9 * (e_afqmc[1] - e_afqmc[4]) - 45 * (e_afqmc[2] - e_afqmc[3]) + e_afqmc[5]) / 60. / dE
obs_afqmc_hf_7p_err = (e_err_afqmc[0]**2 + 9**2 * (e_err_afqmc[1]**2 + e_err_afqmc[4]**2) + 45**2 * (e_err_afqmc[2]**2 + e_err_afqmc[3]**2) + e_err_afqmc[5]**2)**0.5 / 60. / dE

print(f'obs_hf_3p = {(-e_hf[2] + e_hf[3]) / 2. / dE}')
print(f'obs_hf_5p = {(e_hf[1] - 8 * (e_hf[2] - e_hf[3]) - e_hf[4]) / 12. / dE}')
print(f'obs_hf_7p = {(-e_hf[0] + 9 * (e_hf[1] - e_hf[4]) - 45 * (e_hf[2] - e_hf[3]) + e_hf[5]) / 60. / dE}')
print(f'obs_afqmc_hf_3p = {obs_afqmc_hf_3p} +/- {obs_afqmc_hf_3p_err}')
print(f'obs_afqmc_hf_5p = {obs_afqmc_hf_5p} +/- {obs_afqmc_hf_5p_err}')
print(f'obs_afqmc_hf_7p = {obs_afqmc_hf_7p} +/- {obs_afqmc_hf_7p_err}')

