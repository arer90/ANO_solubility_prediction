import os
import numpy as np
import pandas as pd
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit import RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, rdDistGeom, rdPartialCharges
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Descriptors import ExactMolWt

UNSUPPORTED_ELEMENTS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Se', 'Te', 'Rn', 'Xe', 'Kr', 'He', 'Ne', 'Ar'
}

def mol3d(mol):
    mol = Chem.AddHs(mol)
    optimization_methods = [
        (AllChem.EmbedMolecule, (mol, AllChem.ETKDGv3()), {}),
        (AllChem.UFFOptimizeMolecule, (mol,), {'maxIters': 200}),
        (AllChem.MMFFOptimizeMolecule, (mol,), {'maxIters': 200})
    ]

    for method, args, kwargs in optimization_methods:
        try:
            method(*args, **kwargs)
            if mol.GetNumConformers() > 0:
                return mol
        except ValueError as e:
            print(f"Error: {e} - Trying next optimization method [{method}]")

    print(f"Invalid mol for 3d {Chem.MolToSmiles(mol)} - No conformer generated")
    return None

def generating_newfps(fps, descriptor, descriptor_name, save_res="np"):
    try:
        if descriptor is None:
            return fps
            
        if save_res == "pd":
            new_fps = pd.DataFrame(fps) if not isinstance(fps, pd.DataFrame) else fps
            
            if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                try:
                    descriptors_df = pd.DataFrame(
                        {f"{descriptor_name}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
                    )
                    new_fps = pd.concat([new_fps, descriptors_df], axis=1)
                    del descriptor
                except Exception as e:
                    print(f"[-1-] Error occured: {e}")
                    
            elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                try:
                    arrays_1d = [arr[:, None] for arr in descriptor if arr.ndim == 1]
                    arrays_2d = [arr for arr in descriptor if arr.ndim == 2]
                    combined_1d = np.concatenate(arrays_1d, axis=1) if arrays_1d else None
                    combined_2d = np.concatenate(arrays_2d, axis=1) if arrays_2d else None
                    
                    if combined_1d is not None:
                        df_1d = pd.DataFrame(
                            combined_1d,
                            columns=[f'{descriptor_name}_{i+1}' for i in range(combined_1d.shape[1])]
                        )
                        new_fps = pd.concat([new_fps, df_1d], axis=1)
                        
                    if combined_2d is not None:
                        df_2d = pd.DataFrame(
                            combined_2d,
                            columns=[f'{descriptor_name}_{i+1}' for i in range(combined_2d.shape[1])]
                        )
                        new_fps = pd.concat([new_fps, df_2d], axis=1)
                        
                    del descriptor, arrays_1d, arrays_2d
                    if combined_1d is not None: del combined_1d
                    if combined_2d is not None: del combined_2d
                except Exception as e:
                    print(f"[-2-] Error occured: {e}")
                    
            elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                try:
                    descriptor = np.asarray(descriptor).astype('float')
                    descriptors_df = pd.DataFrame(
                        {f"{descriptor_name}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
                    )
                    new_fps = pd.concat([new_fps, descriptors_df], axis=1)
                    del descriptor
                except Exception as e:
                    print(f"[-3-] Error occured: {e}")
                    
            else:
                descriptor = np.asarray(descriptor).astype('float')
                new_fps[descriptor_name] = descriptor.flatten()
                del descriptor
                
            new_fps = new_fps.replace([np.inf, -np.inf], np.nan).fillna(0)
            return new_fps
            
        else:
            new_fps = fps
            
            if descriptor is None:
                pass
            elif isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                try:
                    new_fps = np.concatenate([new_fps, descriptor], axis=1)
                    del descriptor
                except Exception as e:
                    print(f"[-1-] Error occured: {e}")
            elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                try:
                    arrays_1d = [arr[:, None] for arr in descriptor if arr.ndim == 1]
                    arrays_2d = [arr for arr in descriptor if arr.ndim == 2]
                    combined_1d = np.concatenate(arrays_1d, axis=1) if arrays_1d else None
                    combined_2d = np.concatenate(arrays_2d, axis=1) if arrays_2d else None
                    to_concat = [new_fps] + [arr for arr in [combined_1d, combined_2d] if arr is not None]
                    new_fps = np.concatenate(to_concat, axis=1)
                    del descriptor, arrays_1d, arrays_2d
                    if combined_1d is not None: del combined_1d
                    if combined_2d is not None: del combined_2d
                except Exception as e:
                    print(f"[-2-] Error occured: {e}")
            elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                try:
                    descriptor = np.asarray(descriptor).astype('float')
                    new_fps = np.concatenate([new_fps, descriptor], axis=1)
                    del descriptor
                except Exception as e:
                    print(f"[-3-] Error occured: {e}")
            else:
                descriptor = np.asarray(descriptor).astype('float')
                new_fps = np.concatenate([new_fps, descriptor[:,None]], axis=1)
                del descriptor
                
            new_fps = np.nan_to_num(new_fps, nan=0.0, posinf=0.0, neginf=0.0).astype('float')
            return new_fps

    except Exception as e:
        print(f"Error occurred in {descriptor_name}: {e}")
        return fps

def Normalization(descriptor):
    descriptor = np.asarray(descriptor)
    epsilon = 1e-10
    max_value = 1e15
    descriptor = np.clip(descriptor, -max_value, max_value)
    descriptor_custom = np.where(np.abs(descriptor) < epsilon, epsilon, descriptor)
    descriptor_log = np.sign(descriptor_custom) * np.log1p(np.abs(descriptor_custom))
    descriptor_log = np.nan_to_num(descriptor_log, nan=0.0, posinf=0.0, neginf=0.0)
    del epsilon
    gc.collect()    
    return descriptor_log

def values_chi(mol, chi_type):
    i = 0
    chi_func = Chem.GraphDescriptors.ChiNn_ if chi_type == 'n' else Chem.GraphDescriptors.ChiNv_
    while chi_func(mol, i) != 0.0:
        i += 1
    return np.array([chi_func(mol, j) for j in range(i)])

def generate_chi(mols, chi_type):
    n_jobs = os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(values_chi, mol, chi_type) for mol in mols]
        descriptor = [future.result() for future in futures]
    
    max_length = max(len(x) for x in descriptor)
    padded_descriptor = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in descriptor])
    
    return padded_descriptor

# def sanitize_and_compute_descriptor(mol):
#     try:
#         mol = Chem.RemoveHs(mol)
#         Chem.SanitizeMol(mol)
#         try:
#             Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
#         except Exception as e:
#             print(f"Gasteiger charge calculation failed: {e}")
#             return [0] * 8
        
#         try:
#             return Chem.rdMolDescriptors.BCUT2D(mol)
#         except Exception as e:
#             print(f"BCUT2D calculation failed: {e}")
#             return [Descriptors.MolWt(mol)] * 8
#     except Exception as e:
#         print(f"Sanitization failed: {e}")
#         return [0] * 8

def sanitize_and_compute_descriptor(mol):
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception as e:
            print(f"Gasteiger charge calculation failed: {e}")
            return [0] * 8
        
        try:
            return rdMolDescriptors.BCUT2D(mol)
        except Exception as e:
            print(f"BCUT2D calculation failed: {e}")
            return [Descriptors.MolWt(mol)] * 8
    except Exception as e:
        print(f"Sanitization failed: {e}")
        return [0] * 8


# def compute_descriptors_parallel(mols, n_jobs=None):
#     with ProcessPoolExecutor(max_workers=n_jobs) as executor:
#         futures = [executor.submit(sanitize_and_compute_descriptor, mol) for mol in mols if mol is not None]
#         descriptors = [future.result() for future in futures]
#     return np.array(descriptors)

def compute_descriptors_parallel(mols, n_jobs=None):
    """ 병렬로 BCUT2D 및 기타 물성 계산 """
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(sanitize_and_compute_descriptor, mol) for mol in mols if mol is not None]
        descriptors = [future.result() for future in futures]

    # 배열 크기 불일치 해결 (패딩 적용)
    max_length = max(len(d) for d in descriptors)
    padded_descriptors = np.array([np.pad(d, (0, max_length - len(d)), 'constant') for d in descriptors])

    return padded_descriptors

# def process_molecules_parallel(mols, max_workers=4, chunk_size=100):
#     results = []    
#     for i in range(0, len(mols), chunk_size):
#         chunk = mols[i:i + chunk_size]        
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = [executor.submit(mol3d, mol) for mol in chunk]
#             for future in as_completed(futures):
#                 result = future.result()
#                 if result is not None:
#                     results.append(result)        
#         gc.collect()    
#     return results

def process_molecules_parallel(mols, max_workers=4, chunk_size=100):
    """ 병렬로 3D 분자 최적화 수행 (실패한 분자는 None으로 저장) """
    results = [None] * len(mols)  # None으로 초기화하여 길이 유지

    for i in range(0, len(mols), chunk_size):
        chunk = mols[i:i + chunk_size]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(mol3d, mol): idx for idx, mol in enumerate(chunk)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results[i + idx] = result  # 원래 위치 유지 (실패한 분자는 None 유지)

        gc.collect()
    
    return results  # ✅ 이제 valid_indices 없이 결과만 반환


def process_molecules_parallel(mols, max_workers=4, chunk_size=100):
    """ 병렬로 3D 분자 최적화 수행 (실패 시 None으로 패딩) """
    results = [None] * len(mols)  # None으로 초기화하여 길이 유지

    for i in range(0, len(mols), chunk_size):
        chunk = mols[i:i + chunk_size]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(mol3d, mol): idx for idx, mol in enumerate(chunk)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    results[i + idx] = result  # 원래 위치에 저장

        gc.collect()
    
    return [mol for mol in results if mol is not None]  # None 값 제외 후 반환

def search_data_descriptor_compress(trial, fps, mols, name, target_path="result", save_res="np"):
    ####################################
    phase0  = trial.suggest_int("MolWt", 0, 1)                      # 'MolWt'
    phase1  = trial.suggest_int("MolLogP", 0, 1)                    # 'MolLogP'
    phase2  = trial.suggest_int("MolMR", 0, 1)                      # 'MolMR'
    phase3  = trial.suggest_int("TPSA", 0, 1)                       # 'TPSA'
    phase4  = trial.suggest_int("NumRotatableBonds", 0, 1)          # 'NumRotatableBonds'
    phase5  = trial.suggest_int("HeavyAtomCount", 0, 1)             # 'HeavyAtomCount'
    phase6  = trial.suggest_int("NumHAcceptors", 0, 1)              # 'NumHAcceptors'
    phase7  = trial.suggest_int("NumHDonors", 0, 1)                 # 'NumHDonors'
    phase8  = trial.suggest_int("NumHeteroatoms", 0, 1)             # 'NumHeteroatoms'
    phase9  = trial.suggest_int("NumValenceElectrons", 0, 1)        # 'NumValenceElectrons'
    phase10 = trial.suggest_int("NHOHCount", 0, 1)                  # 'NHOHCount'
    phase11 = trial.suggest_int("NOCount", 0, 1)                    # 'NOCount'
    phase12 = trial.suggest_int("RingCount", 0, 1)                  # 'RingCount'
    phase13 = trial.suggest_int("NumAromaticRings", 0, 1)           # 'NumAromaticRings'
    phase14 = trial.suggest_int("NumSaturatedRings", 0, 1)          # 'NumSaturatedRings'
    phase15 = trial.suggest_int("NumAliphaticRings", 0, 1)          # 'NumAliphaticRings'
    phase16 = trial.suggest_int("LabuteASA", 0, 1)                  # 'LabuteASA'
    phase17 = trial.suggest_int("BalabanJ", 0, 1)                   # 'BalabanJ'
    phase18 = trial.suggest_int("BertzCT", 0, 1)                    # 'BertzCT'
    phase19 = trial.suggest_int("Ipc", 0, 1)                        # 'Ipc'
    phase20 = trial.suggest_int("kappa_Series[1-3]_ind", 0, 1)      # 'kappa_Series[1-3]_ind'
    phase21 = trial.suggest_int("Chi_Series[13]_ind", 0, 1)         # 'Chi_Series[13]_ind'
    phase22 = trial.suggest_int("Phi", 0, 1)                        # 'Phi'
    phase23 = trial.suggest_int("HallKierAlpha", 0, 1)              # 'HallKierAlpha'
    phase24 = trial.suggest_int("NumAmideBonds", 0, 1)              # 'NumAmideBonds'
    phase25 = trial.suggest_int("FractionCSP3", 0, 1)               # 'FractionCSP3'
    phase26 = trial.suggest_int("NumSpiroAtoms", 0, 1)              # 'NumSpiroAtoms'
    phase27 = trial.suggest_int("NumBridgeheadAtoms", 0, 1)         # 'NumBridgeheadAtoms'
    phase28 = trial.suggest_int("PEOE_VSA_Series[1-14]_ind", 0, 1)  # 'PEOE_VSA_Series[1-14]_ind'
    phase29 = trial.suggest_int("SMR_VSA_Series[1-10]_ind", 0, 1)   # 'SMR_VSA_Series[1-10]_ind'
    phase30 = trial.suggest_int("SlogP_VSA_Series[1-12]_ind", 0, 1) # 'SlogP_VSA_Series[1-12]_ind'
    phase31 = trial.suggest_int("EState_VSA_Series[1-11]_ind", 0, 1)# 'EState_VSA_Series[1-11]_ind'
    phase32 = trial.suggest_int("VSA_EState_Series[1-10]", 0, 1)    # 'VSA_EState_Series[1-10]'
    phase33 = trial.suggest_int("MQNs", 0, 1)                       # 'MQNs'
    phase34 = trial.suggest_int("AUTOCORR2D", 0, 1)                 # 'AUTOCORR2D'
    phase35 = trial.suggest_int("BCUT2D", 0, 1)                     # 'BCUT2D'
    phase36 = trial.suggest_int("Asphericity", 0, 1)                # 'Asphericity'
    phase37 = trial.suggest_int("PBF", 0, 1)                        # 'PBF'
    phase38 = trial.suggest_int("RadiusOfGyration", 0, 1)           # 'RadiusOfGyration'
    phase39 = trial.suggest_int("InertialShapeFactor", 0, 1)        # 'InertialShapeFactor'
    phase40 = trial.suggest_int("Eccentricity", 0, 1)               # 'Eccentricity'
    phase41 = trial.suggest_int("SpherocityIndex", 0, 1)            # 'SpherocityIndex'
    phase42 = trial.suggest_int("PMI_series[1-3]_ind", 0, 1)        # 'PMI_series[1-3]_ind'
    phase43 = trial.suggest_int("NPR_series[1-2]_ind", 0, 1)        # 'NPR_series[1-2]_ind'
    phase44 = trial.suggest_int("AUTOCORR3D", 0, 1)                 # 'AUTOCORR3D'
    phase45 = trial.suggest_int("RDF", 0, 1)                        # 'RDF'
    phase46 = trial.suggest_int("MORSE", 0, 1)                      # 'MORSE'
    phase47 = trial.suggest_int("WHIM", 0, 1)                       # 'WHIM'
    phase48 = trial.suggest_int("GETAWAY", 0, 1)                    # 'GETAWAY'
    ####################################
    def clear_descriptor_memory(descriptor):
        del descriptor
        gc.collect()
    ####################################
    ####################################
    if phase0 == 1:
        descriptor = [Descriptors.ExactMolWt(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolWt', save_res)
        clear_descriptor_memory(descriptor)
    if phase1 == 1:
        descriptor = [Chem.Crippen.MolLogP(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolLogP', save_res)
        clear_descriptor_memory(descriptor)
    if phase2 == 1:
        descriptor = [Chem.Crippen.MolMR(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolMR', save_res)
        clear_descriptor_memory(descriptor)
    if phase3 == 1:
        descriptor = [Descriptors.TPSA(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'TPSA', save_res)
        clear_descriptor_memory(descriptor)
    if phase4 == 1:
        descriptor = [Chem.Lipinski.NumRotatableBonds(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumRotatableBonds', save_res)
        clear_descriptor_memory(descriptor)
    if phase5 == 1:
        descriptor = [Chem.Lipinski.HeavyAtomCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'HeavyAtomCount', save_res)
        clear_descriptor_memory(descriptor)
    if phase6 == 1:
        descriptor = [Chem.Lipinski.NumHAcceptors(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHAcceptors', save_res)
        clear_descriptor_memory(descriptor)
    if phase7 == 1:
        descriptor = [Chem.Lipinski.NumHDonors(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHDonors', save_res)
        clear_descriptor_memory(descriptor)
    if phase8 == 1:
        descriptor = [Chem.Lipinski.NumHeteroatoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHeteroatoms', save_res)
        clear_descriptor_memory(descriptor)
    if phase9 == 1:
        descriptor = [Chem.Descriptors.NumValenceElectrons(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumValenceElectrons', save_res)
        clear_descriptor_memory(descriptor)
    if phase10 == 1:
        descriptor = [Chem.Lipinski.NHOHCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NHOHCount', save_res)
        clear_descriptor_memory(descriptor)
    if phase11 == 1:
        descriptor = [Chem.Lipinski.NOCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NOCount', save_res)
        clear_descriptor_memory(descriptor)
    if phase12 == 1:
        descriptor = [Chem.Lipinski.RingCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'RingCount', save_res)
        clear_descriptor_memory(descriptor)
    if phase13 == 1:
        descriptor = [Chem.Lipinski.NumAromaticRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAromaticRings', save_res)
        clear_descriptor_memory(descriptor)
    if phase14 == 1:
        descriptor = [Chem.Lipinski.NumSaturatedRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumSaturatedRings', save_res)
        clear_descriptor_memory(descriptor)
    if phase15 == 1:
        descriptor = [Chem.Lipinski.NumAliphaticRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAliphaticRings', save_res)
        clear_descriptor_memory(descriptor)
    if phase16 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcLabuteASA(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'LabuteASA', save_res)
        clear_descriptor_memory(descriptor)
    if phase17 == 1:
        descriptor = [Chem.GraphDescriptors.BalabanJ(alpha) for alpha in mols]
        # descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'BalabanJ', save_res)
        clear_descriptor_memory(descriptor)
    if phase18 == 1:
        descriptor = [Chem.GraphDescriptors.BertzCT(alpha) for alpha in mols]
        # descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'BertzCT', save_res)
        clear_descriptor_memory(descriptor)
    if phase19 == 1:
        descriptor = [Chem.GraphDescriptors.Ipc(alpha) for alpha in mols]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'Ipc', save_res)
        clear_descriptor_memory(descriptor)
    if phase20 == 1:
        d1 = [Chem.GraphDescriptors.Kappa1(alpha) for alpha in mols]
        d2 = [Chem.GraphDescriptors.Kappa2(alpha) for alpha in mols]
        d3 = [Chem.GraphDescriptors.Kappa3(alpha) for alpha in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        fps = generating_newfps(fps, [d1,d2,d3], 'kappa_Series[1-3]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
    if phase21 == 1:
        d1 = [Chem.GraphDescriptors.Chi0(alpha) for alpha in mols]
        d2 = [Chem.GraphDescriptors.Chi0n(alpha) for alpha in mols]
        d3 = [Chem.GraphDescriptors.Chi0v(alpha) for alpha in mols]
        d4 = [Chem.GraphDescriptors.Chi1(alpha) for alpha in mols]
        d5 = [Chem.GraphDescriptors.Chi1n(alpha) for alpha in mols]
        d6 = [Chem.GraphDescriptors.Chi1v(alpha) for alpha in mols]
        d7 = [Chem.GraphDescriptors.Chi2n(alpha) for alpha in mols]
        d8 = [Chem.GraphDescriptors.Chi2v(alpha) for alpha in mols]
        d9 = [Chem.GraphDescriptors.Chi3n(alpha) for alpha in mols]
        d10 = [Chem.GraphDescriptors.Chi3v(alpha) for alpha in mols]
        d11 = [Chem.GraphDescriptors.Chi4n(alpha) for alpha in mols]
        d12 = [Chem.GraphDescriptors.Chi4v(alpha) for alpha in mols]
        d13 = generate_chi(mols, 'n')
        d14 = generate_chi(mols, 'v')
        d1  = np.asarray(d1)
        d2  = np.asarray(d2)
        d3  = np.asarray(d3)
        d4  = np.asarray(d4)
        d5  = np.asarray(d5)
        d6  = np.asarray(d6)
        d7  = np.asarray(d7)
        d8  = np.asarray(d8)
        d9  = np.asarray(d9)
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        d13 = np.asarray(d13)
        d14 = np.asarray(d14)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14], 'Chi_Series[13]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
        clear_descriptor_memory(d12)
        clear_descriptor_memory(d13)
        clear_descriptor_memory(d14)
    if phase22 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcPhi(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'Phi', save_res)
        clear_descriptor_memory(descriptor)
    if phase23 == 1:
        descriptor = [Chem.GraphDescriptors.HallKierAlpha(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'HallKierAlpha', save_res)
        clear_descriptor_memory(descriptor)
    if phase24 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumAmideBonds(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAmideBonds', save_res)
        clear_descriptor_memory(descriptor)
    if phase25 == 1:
        descriptor = [Chem.Lipinski.FractionCSP3(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'FractionCSP3', save_res)
        clear_descriptor_memory(descriptor)
    if phase26 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumSpiroAtoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumSpiroAtoms', save_res)
        clear_descriptor_memory(descriptor)
    if phase27 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumBridgeheadAtoms', save_res)
        clear_descriptor_memory(descriptor)
    if phase28 == 1:
        d1 = [Chem.MolSurf.PEOE_VSA1(alpha) for alpha in mols]
        d2 = [Chem.MolSurf.PEOE_VSA2(alpha) for alpha in mols]
        d3 = [Chem.MolSurf.PEOE_VSA3(alpha) for alpha in mols]
        d4 = [Chem.MolSurf.PEOE_VSA4(alpha) for alpha in mols]
        d5 = [Chem.MolSurf.PEOE_VSA5(alpha) for alpha in mols]
        d6 = [Chem.MolSurf.PEOE_VSA6(alpha) for alpha in mols]
        d7 = [Chem.MolSurf.PEOE_VSA7(alpha) for alpha in mols]
        d8 = [Chem.MolSurf.PEOE_VSA8(alpha) for alpha in mols]
        d9 = [Chem.MolSurf.PEOE_VSA9(alpha) for alpha in mols]
        d10 = [Chem.MolSurf.PEOE_VSA10(alpha) for alpha in mols]
        d11 = [Chem.MolSurf.PEOE_VSA11(alpha) for alpha in mols]
        d12 = [Chem.MolSurf.PEOE_VSA12(alpha) for alpha in mols]
        d13 = [Chem.MolSurf.PEOE_VSA13(alpha) for alpha in mols]
        d14 = [Chem.MolSurf.PEOE_VSA14(alpha) for alpha in mols]
        d1  = np.asarray(d1)
        d2  = np.asarray(d2)
        d3  = np.asarray(d3)
        d4  = np.asarray(d4)
        d5  = np.asarray(d5)
        d6  = np.asarray(d6)
        d7  = np.asarray(d7)
        d8  = np.asarray(d8)
        d9  = np.asarray(d9)
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        d13 = np.asarray(d13)
        d14 = np.asarray(d14)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14],'PEOE_VSA_Series[1-14]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
        clear_descriptor_memory(d12)
        clear_descriptor_memory(d13)
        clear_descriptor_memory(d14)
    if phase29 == 1:
        d1 = [Chem.MolSurf.SMR_VSA1(alpha) for alpha in mols]
        d2 = [Chem.MolSurf.SMR_VSA2(alpha) for alpha in mols]
        d3 = [Chem.MolSurf.SMR_VSA3(alpha) for alpha in mols]
        d4 = [Chem.MolSurf.SMR_VSA4(alpha) for alpha in mols]
        d5 = [Chem.MolSurf.SMR_VSA5(alpha) for alpha in mols]
        d6 = [Chem.MolSurf.SMR_VSA6(alpha) for alpha in mols]
        d7 = [Chem.MolSurf.SMR_VSA7(alpha) for alpha in mols]
        d8 = [Chem.MolSurf.SMR_VSA8(alpha) for alpha in mols]
        d9 = [Chem.MolSurf.SMR_VSA9(alpha) for alpha in mols]
        d10 = [Chem.MolSurf.SMR_VSA10(alpha) for alpha in mols]
        d1  = np.asarray(d1)
        d2  = np.asarray(d2)
        d3  = np.asarray(d3)
        d4  = np.asarray(d4)
        d5  = np.asarray(d5)
        d6  = np.asarray(d6)
        d7  = np.asarray(d7)
        d8  = np.asarray(d8)
        d9  = np.asarray(d9)
        d10 = np.asarray(d10)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10],'SMR_VSA_Series[1-10]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
    if phase30 == 1:
        d1 = [Chem.MolSurf.SlogP_VSA1(alpha) for alpha in mols]
        d2 = [Chem.MolSurf.SlogP_VSA2(alpha) for alpha in mols]
        d3 = [Chem.MolSurf.SlogP_VSA3(alpha) for alpha in mols]
        d4 = [Chem.MolSurf.SlogP_VSA4(alpha) for alpha in mols]
        d5 = [Chem.MolSurf.SlogP_VSA5(alpha) for alpha in mols]
        d6 = [Chem.MolSurf.SlogP_VSA6(alpha) for alpha in mols]
        d7 = [Chem.MolSurf.SlogP_VSA7(alpha) for alpha in mols]
        d8 = [Chem.MolSurf.SlogP_VSA8(alpha) for alpha in mols]
        d9 = [Chem.MolSurf.SlogP_VSA9(alpha) for alpha in mols]
        d10= [Chem.MolSurf.SlogP_VSA10(alpha) for alpha in mols]
        d11= [Chem.MolSurf.SlogP_VSA11(alpha) for alpha in mols]
        d12= [Chem.MolSurf.SlogP_VSA12(alpha) for alpha in mols]
        d1  = np.asarray(d1) 
        d2  = np.asarray(d2) 
        d3  = np.asarray(d3) 
        d4  = np.asarray(d4) 
        d5  = np.asarray(d5) 
        d6  = np.asarray(d6) 
        d7  = np.asarray(d7) 
        d8  = np.asarray(d8) 
        d9  = np.asarray(d9) 
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12],'SlogP_VSA_Series[1-12]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
        clear_descriptor_memory(d12)
    if phase31 == 1:
        d1 = [Chem.EState.EState_VSA.EState_VSA1(alpha) for alpha in mols]
        d2 = [Chem.EState.EState_VSA.EState_VSA2(alpha) for alpha in mols]
        d3 = [Chem.EState.EState_VSA.EState_VSA3(alpha) for alpha in mols]
        d4 = [Chem.EState.EState_VSA.EState_VSA4(alpha) for alpha in mols]
        d5 = [Chem.EState.EState_VSA.EState_VSA5(alpha) for alpha in mols]
        d6 = [Chem.EState.EState_VSA.EState_VSA6(alpha) for alpha in mols]
        d7 = [Chem.EState.EState_VSA.EState_VSA7(alpha) for alpha in mols]
        d8 = [Chem.EState.EState_VSA.EState_VSA8(alpha) for alpha in mols]
        d9 = [Chem.EState.EState_VSA.EState_VSA9(alpha) for alpha in mols]
        d10 = [Chem.EState.EState_VSA.EState_VSA10(alpha) for alpha in mols]
        d11 = [Chem.EState.EState_VSA.EState_VSA11(alpha) for alpha in mols]
        d1  = np.asarray(d1) 
        d2  = np.asarray(d2) 
        d3  = np.asarray(d3) 
        d4  = np.asarray(d4) 
        d5  = np.asarray(d5) 
        d6  = np.asarray(d6) 
        d7  = np.asarray(d7) 
        d8  = np.asarray(d8) 
        d9  = np.asarray(d9) 
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11],'EState_VSA_Series[1-11]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
    if phase32 == 1:
        d1  = [Chem.EState.EState_VSA.VSA_EState1(alpha) for alpha in mols]
        d2  = [Chem.EState.EState_VSA.VSA_EState2(alpha) for alpha in mols]
        d3  = [Chem.EState.EState_VSA.VSA_EState3(alpha) for alpha in mols]
        d4  = [Chem.EState.EState_VSA.VSA_EState4(alpha) for alpha in mols]
        d5  = [Chem.EState.EState_VSA.VSA_EState5(alpha) for alpha in mols]
        d6  = [Chem.EState.EState_VSA.VSA_EState6(alpha) for alpha in mols]
        d7  = [Chem.EState.EState_VSA.VSA_EState7(alpha) for alpha in mols]
        d8  = [Chem.EState.EState_VSA.VSA_EState8(alpha) for alpha in mols]
        d9  = [Chem.EState.EState_VSA.VSA_EState9(alpha) for alpha in mols]
        d10  = [Chem.EState.EState_VSA.VSA_EState10(alpha) for alpha in mols]
        d1  = np.asarray(d1) 
        d2  = np.asarray(d2) 
        d3  = np.asarray(d3) 
        d4  = np.asarray(d4) 
        d5  = np.asarray(d5) 
        d6  = np.asarray(d6) 
        d7  = np.asarray(d7) 
        d8  = np.asarray(d8) 
        d9  = np.asarray(d9) 
        d10 = np.asarray(d10)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10],'VSA_EState_Series[1-10]', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
    if phase33 == 1:
        descriptor = [Chem.rdMolDescriptors.MQNs_(alpha) for alpha in mols]
        # descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'MQNs', save_res)
        clear_descriptor_memory(descriptor)
    if phase34 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcAUTOCORR2D(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'AUTOCORR2D', save_res)
        clear_descriptor_memory(descriptor)
    if phase35 == 1:
        descriptor = compute_descriptors_parallel(mols)
        fps = generating_newfps(fps, descriptor, 'BCUT2D', save_res)
        clear_descriptor_memory(descriptor)
    ####################################################
    mols2 = process_molecules_parallel(mols, max_workers=8)
    del mols
    gc.collect()
    ####################################################
    if phase36 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcAsphericity(alpha) for alpha in mols2]
        fps = generating_newfps(fps, descriptor, 'Asphericity', save_res)
        clear_descriptor_memory(descriptor)
    if phase37 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcPBF(alpha) for alpha in mols2]
        fps = generating_newfps(fps, descriptor, 'PBF', save_res)
        clear_descriptor_memory(descriptor)
    if phase38 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcRadiusOfGyration(alpha) for alpha in mols2]
        fps = generating_newfps(fps, descriptor, 'RadiusOfGyration', save_res)
        clear_descriptor_memory(descriptor)
    if phase39 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcInertialShapeFactor(alpha) for alpha in mols2]
        fps = generating_newfps(fps, descriptor, 'InertialShapeFactor', save_res)
        clear_descriptor_memory(descriptor)
    if phase40 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcEccentricity(alpha) for alpha in mols2]
        fps = generating_newfps(fps, descriptor, 'Eccentricity', save_res)
        clear_descriptor_memory(descriptor)
    if phase41 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcSpherocityIndex(alpha) for alpha in mols2]
        fps = generating_newfps(fps, descriptor, 'SpherocityIndex', save_res)
        clear_descriptor_memory(descriptor)
    if phase42 == 1:
        d1 = [Chem.rdMolDescriptors.CalcPMI1(alpha) for alpha in mols2]
        d2 = [Chem.rdMolDescriptors.CalcPMI2(alpha) for alpha in mols2]
        d3 = [Chem.rdMolDescriptors.CalcPMI3(alpha) for alpha in mols2]
        d1 = Normalization(d1)
        d2 = Normalization(d2)
        d3 = Normalization(d3)
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        fps = generating_newfps(fps, [d1,d2,d3], 'PMI_series[1-3]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
    if phase43 == 1:
        d1 = [Chem.rdMolDescriptors.CalcNPR1(alpha) for alpha in mols2]
        d2 = [Chem.rdMolDescriptors.CalcNPR2(alpha) for alpha in mols2]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        fps = generating_newfps(fps, [d1,d2], 'NPR_series[1-2]_ind', save_res)
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
    if phase44 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcAUTOCORR3D(mols) for mols in mols2]
        fps = generating_newfps(fps, descriptor, 'AUTOCORR3D', save_res)
        clear_descriptor_memory(descriptor)
    if phase45 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcRDF(mols) for mols in mols2]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'RDF', save_res)
        clear_descriptor_memory(descriptor)
    if phase46 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcMORSE(mols) for mols in mols2]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'MORSE', save_res)
        clear_descriptor_memory(descriptor)
    if phase47 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcWHIM(mols) for mols in mols2]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'WHIM', save_res)
        clear_descriptor_memory(descriptor)
    if phase48 == 1:
        descriptor = [Chem.rdMolDescriptors.CalcGETAWAY(mols) for mols in mols2]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'GETAWAY', save_res)
        clear_descriptor_memory(descriptor)
    #########################################
    if save_res == "pd":
        fps.to_csv(f'{target_path}/{name}_feature_selection.csv')
    
    fps = fps.astype('float')
    return fps