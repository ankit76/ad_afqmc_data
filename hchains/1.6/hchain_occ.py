import os
from pyscf import gto, scf, cc, fci
import numpy as np
from functools import partial
import h5py
from scipy.linalg import fractional_matrix_power
from ad_afqmc import driver, pyscf_interface

print = partial(print, flush=True)

r = 1.6
for nH in [ 10, 20, 30, 40, 50, 60, 70 ]:
  print(f'number of H: {nH}\n')
  atomstring = ""
  for i in range(nH):
    atomstring += "H 0 0 %g\n"%(i*r)
  mol = gto.M(atom=atomstring, basis='sto-6g', verbose=3, unit='bohr')
  mf = scf.RHF(mol)
  mf.kernel()

  umf = scf.UHF(mol)
  dm = [np.zeros((mol.nao, mol.nao)), np.zeros((mol.nao, mol.nao))]
  for j in range(mol.nao//2):
    dm[0][2*j, 2*j] = 1.
    dm[1][2*j+1, 2*j+1] = 1.
  umf.kernel(dm)
  mo1 = umf.stability(external=True)[0]
  umf = umf.newton().run(mo1, umf.mo_occ)
  mo1 = umf.stability(external=True)[0]
  umf = umf.newton().run(mo1, umf.mo_occ)

  norb_frozen = 0
  overlap = mf.get_ovlp()
  #lo = fractional_matrix_power(mf.get_ovlp(), -0.5).T
  #lo_mo = umf.mo_coeff[0].T.dot(overlap).dot(lo)
  lo = np.eye(mol.nao)
  lo_mo = umf.mo_coeff[0].T.dot(overlap).dot(lo)
  #h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
  h1 = np.einsum('i,j->ij', lo_mo[:, nH//2], lo_mo[:, nH//2])
  lo_mo_r = mf.mo_coeff.T.dot(overlap).dot(lo)
  h1_r = np.einsum('i,j->ij', lo_mo_r[:, nH//2], lo_mo_r[:, nH//2])

  if nH < 15:
    cisolver = fci.FCI(mf)
    fci_ene, fci_vec = cisolver.kernel()
    print(f'fci_ene: {fci_ene}', flush=True)
    dm1 = cisolver.make_rdm1(fci_vec, mol.nao, mol.nelec)
    np.savetxt(f'rdm1_fci_{nH}.txt', dm1)
    overlap = mf.get_ovlp()
    #h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    print(f'1e ene: {np.trace(np.dot(dm1, h1_r))}')

  pyscf_interface.prep_afqmc(umf, norb_frozen=norb_frozen)
  with h5py.File('observable.h5', 'w') as fh5:
    fh5['constant'] = np.array([ 0. ])
    fh5['op'] = h1.flatten()
  options = {'n_eql': 2,
             'n_ene_blocks': 1,
             'n_sr_blocks': 50,
             'n_blocks': 5,
             'n_walkers': 50,
             'seed': 9889141,
             'walker_type': 'uhf',
             'ad_mode': 'forward'}
  driver.run_afqmc(options=options)
