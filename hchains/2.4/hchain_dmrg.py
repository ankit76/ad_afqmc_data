import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import gto, scf, fci
from scipy.linalg import fractional_matrix_power

r = 2.4
for nH in [ 10, 20, 30, 40, 50, 60, 70 ]:
  print(f'number of H: {nH}\n')
  atomstring = ""
  for i in range(nH):
    atomstring += "H 0 0 %g\n"%(i*r)
  mol = gto.M(atom=atomstring, basis='sto-6g', verbose=3, unit='bohr')
  mf = scf.RHF(mol)
  mf.kernel()

  if nH < 15:
    cisolver = fci.FCI(mf)
    fci_ene, fci_vec = cisolver.kernel()
    print(f'fci_ene: {fci_ene}', flush=True)
    dm1 = cisolver.make_rdm1(fci_vec, mol.nao, mol.nelec)
    np.savetxt('rdm1_fci.txt', dm1)

  lo = fractional_matrix_power(mf.get_ovlp(), -0.5).T
  mo = mf.mo_coeff.copy()
  mf.mo_coeff = lo
  ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
          ncore=0, ncas=None, g2e_symm=8)

  driver = DMRGDriver(scratch="/rc_scratch/anma2640/tmp_dmrg", symm_type=SymmetryTypes.SU2, n_threads=36)
  driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

  bond_dims = [250] * 4 + [500] * 4
  noises = [1e-4] * 4 + [1e-5] * 4 + [0]
  thrds = [1e-10] * 8

  mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
  ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
  energy = driver.dmrg(mpo, ket, n_sweeps=30, bond_dims=bond_dims, noises=noises,
          thrds=thrds, iprint=1)
  print('DMRG energy = %20.15f' % energy)

  pdm1_lo = driver.get_1pdm(ket)
  mo_lo = lo.T.dot(mf.get_ovlp()).dot(mo)
  pdm1_mo = np.einsum('im,jn,ij->mn', mo_lo, mo_lo, pdm1_lo)
  np.savez(f'rdm1_dmrg_{nH}.npz', rdm1=pdm1_mo, basis=mo)
  h1 = mo.T.dot(mf.get_hcore()).dot(mo)
  print(f"dmrg observable: {np.einsum('ij,ij->', pdm1_mo, h1)}")
