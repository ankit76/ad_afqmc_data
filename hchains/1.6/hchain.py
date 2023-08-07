import os
from pyscf import gto, scf, cc, fci
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
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

  if nH < 15:
    cisolver = fci.FCI(mf)
    fci_ene, fci_vec = cisolver.kernel()
    print(f'fci_ene: {fci_ene}', flush=True)
    dm1 = cisolver.make_rdm1(fci_vec, mol.nao, mol.nelec)
    np.savetxt(f'rdm1_fci_{nH}.txt', dm1)
    overlap = mf.get_ovlp()
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    print(f'1e ene: {np.trace(np.dot(dm1, h1))}')

  # ccsd
  mycc = cc.CCSD(mf)
  mycc.frozen = norb_frozen
  mycc.kernel()
  dm1_cc = mycc.make_rdm1()
  np.savez(f'rdm1_ccsd_{nH}.npz', rdm1=dm1_cc, basis=mf.mo_coeff)

  eris = mycc.ao2mo()
  conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)
  dm1_ccsd_t = ccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2, eris=eris)
  np.savez(f'rdm1_ccsd_t_{nH}.npz', rdm1=dm1_ccsd_t, basis=mf.mo_coeff)

  # uccsd
  mycc = cc.CCSD(umf)
  mycc.frozen = norb_frozen
  mycc.kernel()
  dm1_cc = mycc.make_rdm1()
  np.savez(f'rdm1_uccsd_{nH}.npz', rdm1=dm1_cc, basis=umf.mo_coeff)

  eris = mycc.ao2mo()
  conv, l1, l2 = uccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)
  dm1_ccsd_t = uccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2, eris=eris)
  np.savez(f'rdm1_uccsd_t_{nH}.npz', rdm1=dm1_ccsd_t, basis=umf.mo_coeff)

  pyscf_interface.finite_difference_properties(mol, mf.get_hcore(), norb_frozen=norb_frozen, relaxed=True)
  pyscf_interface.finite_difference_properties(mol, mf.get_hcore(), norb_frozen=norb_frozen, hf_type='uhf', relaxed=True, dm=dm)

  # afqmc
  pyscf_interface.prep_afqmc(umf, norb_frozen=norb_frozen)
  options = {'n_eql': 2,
             'n_ene_blocks': 1,
             'n_sr_blocks': 50,
             'n_blocks': 5,
             'n_walkers': 50,
             'seed': 9889141,
             'walker_type': 'uhf',
             'ad_mode': 'reverse'}
  driver.run_afqmc(options=options)

  rdm1_afqmc = np.load('rdm1_afqmc.npz')['rdm1']
  np.savez(f'rdm1_afqmc_{nH}.npz', rdm1=rdm1_afqmc, basis=umf.mo_coeff[0])
  os.system(f'rm rdm1_afqmc.npz -f; mv samples.dat samples_{nH}.dat; mv samples_raw.dat samples_raw_{nH}.dat')
