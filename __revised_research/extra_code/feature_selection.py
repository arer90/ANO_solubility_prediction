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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import optuna

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

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Union, List, Optional

def process_chunk_optimized(chunk_data):
    chunk, name_prefix, start_idx = chunk_data
    return pd.DataFrame(
        chunk,
        columns=[f"{name_prefix}_{j+1}" for j in range(start_idx, start_idx + chunk.shape[1])]
    )

def generate_df_concurrently(descriptor: np.ndarray, name_prefix: str, chunk_size: int = 1000) -> Optional[pd.DataFrame]:
    try:
        chunks = [
            (descriptor[:, i:min(i + chunk_size, descriptor.shape[1])], name_prefix, i)
            for i in range(0, descriptor.shape[1], chunk_size)
        ]
        
        with ProcessPoolExecutor() as executor:
            chunk_dfs = list(executor.map(process_chunk_optimized, chunks))
            
        return pd.concat(chunk_dfs, axis=1) if chunk_dfs else None
        
    except Exception as e:
        print(f"[-1-] Error in generating DataFrame concurrently: {e}")
        return pd.DataFrame(
            {f"{name_prefix}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
        )

def generating_newfps(
    fps: Union[np.ndarray, pd.DataFrame],
    descriptor: Optional[Union[np.ndarray, List[np.ndarray], List[List]]],
    descriptor_name: str,
    save_res: str = "np",
    chunk_size: int = 1000
) -> Union[np.ndarray, pd.DataFrame]:
    try:
        if descriptor is None:
            return fps

        if save_res == "pd":
            new_fps = pd.DataFrame(fps) if not isinstance(fps, pd.DataFrame) else fps

            if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                descriptors_df = generate_df_concurrently(descriptor, descriptor_name, chunk_size)
                if descriptors_df is not None:
                    new_fps = pd.concat([new_fps, descriptors_df], axis=1)

            elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                try:
                    combined = np.hstack([
                        arr if arr.ndim > 1 else arr.reshape(-1, 1)
                        for arr in descriptor
                    ])
                    descriptors_df = generate_df_concurrently(combined, descriptor_name, chunk_size)
                    if descriptors_df is not None:
                        new_fps = pd.concat([new_fps, descriptors_df], axis=1)
                except Exception as e:
                    print(f"[-2-] Error processing array list: {e}")

            elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                try:
                    descriptor_array = np.asarray(descriptor, dtype=np.float32)
                    descriptors_df = generate_df_concurrently(descriptor_array, descriptor_name, chunk_size)
                    if descriptors_df is not None:
                        new_fps = pd.concat([new_fps, descriptors_df], axis=1)
                except Exception as e:
                    print(f"[-3-] Error processing nested list: {e}")

            else:
                try:
                    descriptor_array = np.asarray(descriptor, dtype=np.float32)
                    new_fps[descriptor_name] = descriptor_array.flatten()
                except Exception as e:
                    print(f"[-4-] Error processing single descriptor: {e}")

            new_fps.replace([np.inf, -np.inf], np.nan, inplace=True)
            new_fps.fillna(0, inplace=True)
            return new_fps

        else:  # numpy 처리
            try:
                if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                    new_fps = np.concatenate([fps, descriptor], axis=1)
                elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                    combined_arrays = [
                        arr if arr.ndim > 1 else arr.reshape(-1, 1)
                        for arr in descriptor
                    ]
                    new_fps = np.concatenate([fps] + combined_arrays, axis=1)
                elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                    descriptor_array = np.asarray(descriptor, dtype=np.float32)
                    new_fps = np.concatenate([fps, descriptor_array], axis=1)
                else:
                    descriptor_array = np.asarray(descriptor, dtype=np.float32)
                    new_fps = np.concatenate([fps, descriptor_array[:, None]], axis=1)

                return np.nan_to_num(new_fps, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
            except Exception as e:
                print(f"[-5-] Error in numpy processing: {e}")
                return fps

    except Exception as e:
        print(f"[-6-] General error in {descriptor_name}: {e}")
        return fps

# def generating_newfps(fps, descriptor, descriptor_name, save_res="np"):
#     try:
#         if descriptor is None:
#             return fps
            
#         if save_res == "pd":
#             new_fps = pd.DataFrame(fps) if not isinstance(fps, pd.DataFrame) else fps
            
#             if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
#                 try:
#                     descriptors_df = pd.DataFrame(
#                         {f"{descriptor_name}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
#                     )
#                     new_fps = pd.concat([new_fps, descriptors_df], axis=1)
#                     del descriptor
#                 except Exception as e:
#                     print(f"[-1-] Error occured: {e}")
                    
#             elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
#                 try:
#                     arrays_1d = [arr[:, None] for arr in descriptor if arr.ndim == 1]
#                     arrays_2d = [arr for arr in descriptor if arr.ndim == 2]
#                     combined_1d = np.concatenate(arrays_1d, axis=1) if arrays_1d else None
#                     combined_2d = np.concatenate(arrays_2d, axis=1) if arrays_2d else None
                    
#                     if combined_1d is not None:
#                         df_1d = pd.DataFrame(
#                             combined_1d,
#                             columns=[f'{descriptor_name}_{i+1}' for i in range(combined_1d.shape[1])]
#                         )
#                         new_fps = pd.concat([new_fps, df_1d], axis=1)
                        
#                     if combined_2d is not None:
#                         df_2d = pd.DataFrame(
#                             combined_2d,
#                             columns=[f'{descriptor_name}_{i+1}' for i in range(combined_2d.shape[1])]
#                         )
#                         new_fps = pd.concat([new_fps, df_2d], axis=1)
                        
#                     del descriptor, arrays_1d, arrays_2d
#                     if combined_1d is not None: del combined_1d
#                     if combined_2d is not None: del combined_2d
#                 except Exception as e:
#                     print(f"[-2-] Error occured: {e}")
                    
#             elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
#                 try:
#                     descriptor = np.asarray(descriptor).astype('float')
#                     descriptors_df = pd.DataFrame(
#                         {f"{descriptor_name}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
#                     )
#                     new_fps = pd.concat([new_fps, descriptors_df], axis=1)
#                     del descriptor
#                 except Exception as e:
#                     print(f"[-3-] Error occured: {e}")
                    
#             else:
#                 descriptor = np.asarray(descriptor).astype('float')
#                 new_fps[descriptor_name] = descriptor.flatten()
#                 del descriptor
                
#             new_fps = new_fps.replace([np.inf, -np.inf], np.nan).fillna(0)
#             return new_fps
            
#         else:
#             new_fps = fps
            
#             if descriptor is None:
#                 pass
#             elif isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
#                 try:
#                     new_fps = np.concatenate([new_fps, descriptor], axis=1)
#                     del descriptor
#                 except Exception as e:
#                     print(f"[-1-] Error occured: {e}")
#             elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
#                 try:
#                     arrays_1d = [arr[:, None] for arr in descriptor if arr.ndim == 1]
#                     arrays_2d = [arr for arr in descriptor if arr.ndim == 2]
#                     combined_1d = np.concatenate(arrays_1d, axis=1) if arrays_1d else None
#                     combined_2d = np.concatenate(arrays_2d, axis=1) if arrays_2d else None
#                     to_concat = [new_fps] + [arr for arr in [combined_1d, combined_2d] if arr is not None]
#                     new_fps = np.concatenate(to_concat, axis=1)
#                     del descriptor, arrays_1d, arrays_2d
#                     if combined_1d is not None: del combined_1d
#                     if combined_2d is not None: del combined_2d
#                 except Exception as e:
#                     print(f"[-2-] Error occured: {e}")
#             elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
#                 try:
#                     descriptor = np.asarray(descriptor).astype('float')
#                     new_fps = np.concatenate([new_fps, descriptor], axis=1)
#                     del descriptor
#                 except Exception as e:
#                     print(f"[-3-] Error occured: {e}")
#             else:
#                 descriptor = np.asarray(descriptor).astype('float')
#                 new_fps = np.concatenate([new_fps, descriptor[:,None]], axis=1)
#                 del descriptor
                
#             new_fps = np.nan_to_num(new_fps, nan=0.0, posinf=0.0, neginf=0.0).astype('float')
#             return new_fps

#     except Exception as e:
#         print(f"Error occurred in {descriptor_name}: {e}")
#         return fps  

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

def sanitize_and_compute_descriptor(mol):
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception as e:
            print(f"Gasteiger charge calculation failed: {e}")
            return [0] * 8
        
        try:
            return Chem.rdMolDescriptors.BCUT2D(mol)
        except Exception as e:
            print(f"BCUT2D calculation failed: {e}")
            return [Descriptors.MolWt(mol)] * 8
    except Exception as e:
        return [0] * 8

def compute_descriptors_parallel(mols, n_jobs=None):
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(sanitize_and_compute_descriptor, mol) for mol in mols if mol is not None]
        descriptors = [future.result() for future in futures]
    return np.array(descriptors)

def process_molecules_parallel(mols, max_workers=4, chunk_size=100):
    results = []    
    for i in range(0, len(mols), chunk_size):
        chunk = mols[i:i + chunk_size]        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(mol3d, mol) for mol in chunk]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)        
        gc.collect()    
    return results

def selection_data_descriptor_compress(selection, fps, mols, name, target_path="result", save_res="np"):
    if save_res == "pd":
        fps = pd.DataFrame({'mols': mols})
    ####################################
    phase0  = 1 #selection[0]  #"MolWeight" #
    phase1  = 1 #selection[1]  #"Mol_logP"  #
    phase2  = 1 #selection[2]  #"Mol_MR"    #
    phase3  = 1 #selection[3]  #"Mol_TPSA"  #
    phase4  = selection[4]  #"NumRotatableBonds" #
    phase5  = selection[5]  #"HeavyAtomCount" #
    phase6  = selection[6]  #"NumHAcceptors"  #
    phase7  = selection[7]  #"NumHDonors"     #
    phase8  = selection[8]  #"NumHeteroatoms" #
    phase9  = selection[9]  #"NumValenceElec" #
    phase10 = selection[10] #"NHOHCount"      #
    phase11 = selection[11] #"NOCount"        # 
    phase12 = selection[12] #"RingCount"      #
    phase13 = selection[13] #"NumAromaticRings"  #
    phase14 = selection[14] #"NumSaturatedRings" #
    phase15 = selection[15] #"NumAliphaticRings" #
    phase16 = selection[16] #"LabuteASA" #
    phase17 = selection[17] #"BalabanJ"  #
    phase18 = selection[18] #"BertzCT"   #
    phase19 = selection[19] #"Ipc"       #
    phase20 = selection[20] #"kappa_Series[1-3]_ind" #
    phase21 = selection[21] #"Chi_Series[13]_ind"    #
    phase22 = selection[22] #"Phi"                   #
    phase23 = selection[23] #"HallKierAlpha"         #
    phase24 = selection[24] #"NumAmideBonds"         #
    phase25 = selection[25] #"FractionCSP3"          #
    phase26 = selection[26] #"NumSpiroAtoms"         #
    phase27 = selection[27] #"NumBridgeheadAtoms"    #
    phase28 = selection[28] #"PEOE_VSA_Series[1-14]_ind" #
    phase29 = selection[29] #"SMR_VSA_Series[1-10]_ind"  #
    phase30 = selection[30] #"SlogP_VSA_Series[1-12]_ind"# 
    phase31 = selection[31] #"EState_VSA_Series[1-11]_ind"#
    phase32 = selection[32] #"VSA_EState_Series[1-10]_ind"#
    phase33 = selection[33] #"MQNs"              #
    phase34 = selection[34] #"AUTOCORR2D"        #
    phase35 = selection[35] #"BCUT2D"            #
    phase36 = selection[36] #"Asphericity"       #
    phase37 = selection[37] #"PBF"               #
    phase38 = selection[38] #"RadiusOfGyration"  #
    phase39 = selection[39] #"InertialShapeFactor"#
    phase40 = selection[40] #"Eccentricity"
    phase41 = selection[41] #"SpherocityIndex"
    phase42 = selection[42] #"PMI_series[1-3]_ind"
    phase43 = selection[43] #"NPR_series[1-2]_ind"
    phase44 = selection[44] #"AUTOCORR3D"
    phase45 = selection[45] #"RDF"
    phase46 = selection[46] #"MORSE"
    phase47 = selection[47] #"WHIM"
    phase48 = selection[48] #"GETAWAY"
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


def selection_fromStudy_compress(study_name, storage, unfixed=False, showlog=True):
    model_fea = np.zeros(49, dtype=int)
    study = optuna.load_study(study_name=study_name, storage=storage)

    best_trial = study.best_trial
    
    required_features = ["MolWt", "MolLogP", "MolMR", "TPSA"]
    required_indices = [0, 1, 2, 3]

    param_to_index = {
        "MolWt": 0,
        "MolLogP": 1,
        "MolMR": 2,
        "TPSA": 3,
        "NumRotatableBonds": 4,
        "HeavyAtomCount": 5,
        "NumHAcceptors": 6,
        "NumHDonors": 7,
        "NumHeteroatoms": 8,
        "NumValenceElectrons": 9,
        "NHOHCount": 10,
        "NOCount": 11,
        "RingCount": 12,
        "NumAromaticRings": 13,
        "NumSaturatedRings": 14,
        "NumAliphaticRings": 15,
        "LabuteASA": 16,
        "BalabanJ": 17,
        "BertzCT": 18,
        "Ipc": 19,
        "kappa_Series[1-3]_ind": 20,
        "Chi_Series[13]_ind": 21,
        "Phi": 22,
        "HallKierAlpha": 23,
        "NumAmideBonds": 24,
        "FractionCSP3": 25,
        "NumSpiroAtoms": 26,
        "NumBridgeheadAtoms": 27,
        "PEOE_VSA_Series[1-14]_ind": 28,
        "SMR_VSA_Series[1-10]_ind": 29,
        "SlogP_VSA_Series[1-12]_ind": 30,
        "EState_VSA_Series[1-11]_ind": 31,
        "VSA_EState_Series[1-10]": 32,
        "MQNs": 33,
        "AUTOCORR2D": 34,
        "BCUT2D": 35,
        "Asphericity": 36,
        "PBF": 37,
        "RadiusOfGyration": 38,
        "InertialShapeFactor": 39,
        "Eccentricity": 40,
        "SpherocityIndex": 41,
        "PMI_series[1-3]_ind": 42,
        "NPR_series[1-2]_ind": 43,
        "AUTOCORR3D": 44,
        "RDF": 45,
        "MORSE": 46,
        "WHIM": 47,
        "GETAWAY": 48
    }

    if not unfixed:
        model_fea[required_indices] = 1
        
        for param in best_trial.params:
            if param in param_to_index and param not in required_features:
                model_fea[param_to_index[param]] = best_trial.params[param]
    else:
        for param in best_trial.params:
            if param in param_to_index:
                model_fea[param_to_index[param]] = best_trial.params[param]

    if showlog:
        print(f"Best trial for study '{study_name}':")
        print("Best trial value:", best_trial.value)
        print("Best trial parameters:", best_trial.params)
        print("Generated fea:", model_fea)
        if not unfixed:
            print("Fixed features:", required_features)

    return model_fea

def selection_structure_compress(study_name, storage, input_dim, returnOnly=False):
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    print("Best trial params:", best_trial.params)
    
    try:
        lr = best_trial.params["lr"]
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Error occurred: changing name 'lr' to 'Learning_rate'")
        lr = best_trial.params["Learning_rate"]
    
    if returnOnly:
        return lr
    
    n_layers = best_trial.params["n_layers"]
    model = tf.keras.Sequential()
    layer_dropout = best_trial.params["layer_dropout"]
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    
    for i in range(n_layers):
        num_hidden = best_trial.params[f"n_units_l_{i}"]
        num_decay = best_trial.params[f"n_decay_l_{i}"]
        
        model.add(tf.keras.layers.Dense(
            num_hidden,
            activation="relu",
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(num_decay),
        ))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        if layer_dropout == 1:
            fdropout = best_trial.params[f"F_dropout_{i}"]
            model.add(tf.keras.layers.Dropout(rate=fdropout))
            
    if layer_dropout == 0:
        final_dropout = best_trial.params["last_dropout"]
        model.add(tf.keras.layers.Dropout(rate=final_dropout))
        
    model.add(tf.keras.layers.Dense(units=1))
    
    print(f"Model created from best trial of '{study_name}':")
    print("  Params:", best_trial.params)
    print("  Best trial value:", best_trial.value)
    return model, lr