import os
from pyscf import gto, scf, cc, fci
import numpy as np
from functools import partial
import h5py
from scipy.linalg import fractional_matrix_power
from ad_afqmc import driver, pyscf_interface

print = partial(print, flush=True)

r = 2.4
nH = 20
atomstring = ""
for i in range(nH):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(atom=atomstring, basis='sto-6g', verbose=3, unit='bohr')
mf = scf.RHF(mol)
mf.kernel()

umf = scf.UHF(mol)
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)

norb_frozen = 0
# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = norb_frozen
mycc.kernel()
et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)

mycc = cc.CCSD(umf)
mycc.frozen = norb_frozen
mycc.kernel()
et = mycc.ccsd_t()
print('UCCSD(T) energy', mycc.e_tot + et)

pyscf_interface.finite_difference_properties(mol, mf.get_hcore(), norb_frozen=norb_frozen)
pyscf_interface.finite_difference_properties(mol, mf.get_hcore(), norb_frozen=norb_frozen, relaxed=False)
pyscf_interface.finite_difference_properties(mol, mf.get_hcore(), norb_frozen=norb_frozen, hf_type='uhf')
pyscf_interface.finite_difference_properties(mol, mf.get_hcore(), norb_frozen=norb_frozen, hf_type='uhf', relaxed=False)

pyscf_interface.prep_afqmc(umf, norb_frozen=norb_frozen)

options = {'n_eql': 2,
           'n_ene_blocks': 1,
           'n_blocks': 40,
           'n_walkers': 50,
           'seed': 98,
           'walker_type': 'uhf',
           'orbital_rotation': True,
           'ad_mode': 'forward'}

for n_sr_blocks in [ 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 ]:
  print(f'\nn_sr_blocks: {n_sr_blocks}\n')
  options['n_sr_blocks'] = n_sr_blocks
  driver.run_afqmc(options=options)

