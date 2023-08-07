import os
from pyscf import gto, scf, cc, fci
import numpy as np
from functools import partial
import h5py
from scipy.linalg import fractional_matrix_power
from ad_afqmc import driver, pyscf_interface

print = partial(print, flush=True)

atomstring = f'''
C 0.00000000 2.11924634 0.62676569
C 0.00000000 -2.11924634 0.62676569
C 0.00000000 1.34568862 -1.85506908
C 0.00000000 -1.34568862 -1.85506908
N 0.00000000 0.00000000 2.10934391
H 0.00000000 0.00000000 4.00257355
H 0.00000000 3.97648410 1.44830201
H 0.00000000 -3.97648410 1.44830201
H 0.00000000 2.56726559 -3.47837232
H 0.00000000 -2.56726559 -3.47837232
'''
mol = gto.M(atom=atomstring, basis='augccpvtz', verbose=3, symmetry=0, unit='bohr')
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 5
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

pyscf_interface.finite_difference_properties(mol, dip_ints_ao[2], observable_constant=nuc_dipmom[2],norb_frozen=norb_frozen)
pyscf_interface.finite_difference_properties(mol, dip_ints_ao[2], observable_constant=nuc_dipmom[2], norb_frozen=norb_frozen, relaxed=False)

pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen, chol_cut=1.e-5)
# frozen orbitals
dip_ints_mo_act = np.zeros((dip_ints_ao.shape[0], mol.nao - norb_frozen, mol.nao - norb_frozen))
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo_act[i] = dip_ints_mo[i][norb_frozen:, norb_frozen:]
  nuc_dipmom[i] += 2. * np.trace(dip_ints_mo[i][:norb_frozen, :norb_frozen])
dip_ints_mo = dip_ints_mo_act
with h5py.File('observable.h5', 'w') as fh5:
    fh5['constant'] = np.array([ nuc_dipmom[2] ])
    fh5['op'] = dip_ints_mo[2].flatten()

options = {'n_eql': 2,
           'n_ene_blocks': 1,
           'n_sr_blocks': 50,
           'n_blocks': 8,
           'n_walkers': 50,
           'seed': 1498,
           'orbital_rotation': True,
           'ad_mode': 'forward'}
driver.run_afqmc(options=options, nproc=12)

