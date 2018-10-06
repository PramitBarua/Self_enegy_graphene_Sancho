#!/usr/bin/env python3

__author__ = "Pramit Barua"
__copyright__ = "Copyright 2018, INT, KIT"
__credits__ = ["Pramit Barua"]
__license__ = "INT, KIT"
__version__ = "1"
__maintainer__ = "Pramit Barua"
__email__ = ["pramit.barua@student.kit.edu", "pramit.barua@gmail.com"]

'''

'''
import time
import argparse
import numpy as np
# import matplotlib.pyplot as plt 


from NEGF_package import NEGFGlobal
from NEGF_package import GenerateHamiltonian
from NEGF_package import Alpha

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("Folder_Name",
                        help="address of the folder that contains 'main_control_parameter.yml' file")
    args = parser.parse_args()

    input_parameter = NEGFGlobal.yaml_file_loader(args.Folder_Name,
                                                  'main_control_parameter.yml')

    location = input_parameter['File_location']

    message = ['=== Program Start ===']
    NEGFGlobal.global_write(location, 'output.out', message=message)

    a = 1.42
    ax = np.sqrt(3)*a
    bz = 3*a
    eta = 0.001
    NK = 1000
    NE = 100
    start_time = time.time()

    file_name = 'coordinate_' + input_parameter['system_name'] + '.ao'

    ks_matrix, overlap_matrix = NEGFGlobal.ao_file_loader(location, file_name)

#     NEGFGlobal.global_write(location, 'ks_matrix.dat', num_data=ks_matrix)
#     NEGFGlobal.global_write(location, 'overlap_matrix.dat', num_data=overlap_matrix)

    file_name = 'map_' + input_parameter['system_name'] + '.csv'
    map_file = NEGFGlobal.csv_file_loader(location, file_name)

    file_name = 'map_coordinate_' + input_parameter['system_name'] + '.csv'
    map_coordinate_file = NEGFGlobal.csv_file_loader(location, file_name)

    center_ss = input_parameter['Center_ss']
    kz = np.linspace(-np.pi/bz, np.pi/bz, NK)

    # top part
    num_unit_cell_top = [int(item) for item in input_parameter['Num_unit_cell']['Top_part'].split(', ')]
 
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_top):
        num_unit_cell = np.array([input_parameter['Center_ss'], item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)
 
        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)
 
    Hsr = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    Ssr = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    # middle part
    num_unit_cell_middle = [int(item) for item in input_parameter['Num_unit_cell']['Middle_part'].split(', ')]

    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_middle):
        num_unit_cell = np.array([input_parameter['Center_ss'], item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)

        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)

    Hs = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    Ss = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    # bottom part
    num_unit_cell_bottom = [int(item) for item in input_parameter['Num_unit_cell']['Bottom_part'].split(', ')]
 
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_bottom):
        num_unit_cell = np.array([input_parameter['Center_ss'], item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)
 
        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)
 
    Hls = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    Sls = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    E = np.linspace(-10, 10, NE)
    E = E + 1j*eta

    Hs_shape = Hs.shape
    Gka = np.zeros(NE, dtype = complex)#sum
    for idE, energy_item in enumerate(E):
#         print(idE)
        Gks = 0
        for idk, item_k in enumerate(kz):
            gl = Alpha.self_energy(energy_item, Hs[idk], Hls[idk], Ss[idk], Sls[idk])
            sigma_l = (energy_item*Sls[idk] - Hls[idk]) @ gl @ np.matrix.getH(energy_item*Sls[idk] - Hls[idk])

            gr = Alpha.self_energy(energy_item, Hs[idk], Hsr[idk], Ss[idk], Ssr[idk])
            sigma_r = (energy_item*Ssr[idk] - Hsr[idk]) @ gr @ np.matrix.getH(energy_item*Ssr[idk] - Hsr[idk])

            Gks += np.linalg.inv(energy_item*Ss[idk] - Hs[idk] - sigma_l - sigma_r)

        Gka[idE] = np.trace(Gks)/NK#average
    DOS = -1.0/np.pi * np.imag(Gka)

    E = E - (-0.08084022131023*27.2114)
    NEGFGlobal.display(location, 'DOS.png', E, DOS, xlabel='Energy', ylabel='DOS')
