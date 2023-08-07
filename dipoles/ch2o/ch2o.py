import os
from pyscf import gto, scf, cc, fci
import numpy as np
from functools import partial
import h5py
from scipy.linalg import fractional_matrix_power
from ad_afqmc import driver, pyscf_interface

print = partial(print, flush=True)

atomstring = f'''
                 C 0.00000000 1.13947666  0.00000000
                 O 0.00000000 -1.14402883 0.00000000
                 H 0.00000000 2.23398653  1.76627623
                 H 0.00000000 2.23398653 -1.76627623
              '''
mol = gto.M(atom=atomstring, basis='augccpvtz', verbose=3, symmetry=0, unit='bohr')
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 2
overlap = mf.get_ovlp()
h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)

# dipole moment
nuc_dipmom = [0.0, 0.0, 0.0]
for i in range(mol.natm):
  for j in range(3):
    nuc_dipmom[j] += mol.atom_charge(i) * mol.atom_coord(i)[j]

# spatial orbitals
dip_ints_ao = -mol.intor_symmetric('int1e_r', comp=3)
dip_ints_mo = np.empty_like(dip_ints_ao)
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo[i] = mf.mo_coeff.T.dot(dip_ints_ao[i]).dot(mf.mo_coeff)

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = norb_frozen
mycc.kernel()
dm1_cc = mycc.make_rdm1()

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)

edip_cc = [0., 0. ,0.]
for i in range(3):
  edip_cc[i] = np.trace(np.dot(dm1_cc, dip_ints_mo[i]))
print(f'ccsd dipole (unrelaxed): {np.array(nuc_dipmom) + np.array(edip_cc)}')

pyscf_interface.finite_difference_properties(mol, dip_ints_ao[1], observable_constant=nuc_dipmom[1], norb_frozen=norb_frozen)
pyscf_interface.finite_difference_properties(mol, dip_ints_ao[1], observable_constant=nuc_dipmom[1], norb_frozen=norb_frozen, relaxed=False)
exit()

pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen)
# frozen orbitals
dip_ints_mo_act = np.zeros((dip_ints_ao.shape[0], mol.nao - norb_frozen, mol.nao - norb_frozen))
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo_act[i] = dip_ints_mo[i][norb_frozen:, norb_frozen:]
  nuc_dipmom[i] += 2. * np.trace(dip_ints_mo[i][:norb_frozen, :norb_frozen])
dip_ints_mo = dip_ints_mo_act
with h5py.File('observable.h5', 'w') as fh5:
    fh5['constant'] = np.array([ nuc_dipmom[1] ])
    fh5['op'] = dip_ints_mo[1].flatten()

options = {'n_eql': 2,
           'n_ene_blocks': 1,
           'n_sr_blocks': 50,
           'n_blocks': 4,
           'n_walkers': 50,
           'seed': 98,
           'orbital_rotation': True,
           'ad_mode': 'reverse'}
driver.run_afqmc(options=options)

options['orbital_rotation'] = False

for i in range(3):
  print(f'scf iteration {i}:\n')
  afqmc_rdm = np.load('rdm1_afqmc.npz')['rdm1']
  occ, no = np.linalg.eigh(afqmc_rdm)
  no_ao = mf.mo_coeff[:, norb_frozen:].dot(no)
  mf.mo_coeff[:, norb_frozen:]  = no_ao[:, ::-1]
  np.savez(f'mo_coeff_{i}.npz', mf.mo_coeff)
  pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen)

  dip_ints_mo[1] = no[:, ::-1].T.dot(dip_ints_mo[1]).dot(no[:, ::-1])
  with h5py.File('observable.h5', 'w') as fh5:
      fh5['constant'] = np.array([ nuc_dipmom[1] ])
      fh5['op'] = dip_ints_mo[1].flatten()

  driver.run_afqmc(options=options)

