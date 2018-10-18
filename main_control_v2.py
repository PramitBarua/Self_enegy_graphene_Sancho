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
import matplotlib.pyplot as plt 


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
    NK = 100
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

#     center_ss = input_parameter['Center_graphene']
    kz = np.linspace(-np.pi/bz, np.pi/bz, NK)

    # top part
    num_unit_cell_top = [int(item) for item in input_parameter['Num_unit_cell']['Top_part'].split(', ')]
 
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_top):
        num_unit_cell = np.array([input_parameter['Center_graphene'], item], dtype=int)
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
        num_unit_cell = np.array([input_parameter['Center_graphene'], item], dtype=int)
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
        num_unit_cell = np.array([input_parameter['Center_graphene'], item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)
 
        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)
 
    Hls = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    Sls = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    # tube part
    num_unit_cell_tube = [int(item) for item in input_parameter['Num_unit_cell']['tube_part'].split(', ')]

    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_tube):
        num_unit_cell = np.array([input_parameter['Center_tube'], item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)

        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)

    H_tube = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    S_tube = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    # graphene tube
    try:
        num_unit_cell_graphene = [int(item) for item in input_parameter['tube_graphene_interaction']['graphene_part'].split(', ')]
    except AttributeError:
        num_unit_cell_graphene = [input_parameter['tube_graphene_interaction']['graphene_part']]

    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((len(num_unit_cell_graphene),3), dtype = float)
    for id, item in enumerate(num_unit_cell_graphene):
        num_unit_cell = np.array([input_parameter['Center_tube'], item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)

        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)

    H_beta = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    S_beta = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    #graphene graphene
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((1,3), dtype = float)

    num_unit_cell = np.array([input_parameter['Center_graphene'], input_parameter['Center_graphene']], dtype=int)
    hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
    hamiltonian_block.append(hamiltonian)

    overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
    overlap_block.append(overlap)

    H_graphene = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    S_graphene = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    E = np.linspace(-1-0.07894735985743*27.2114, 1-0.07894735985743*27.2114, NE)
    E = E + 1j*eta

    tube_shape = H_tube.shape
    sigma11 = np.zeros((NE, tube_shape[1], tube_shape[2]), dtype=complex)
    for idE, energy_item in enumerate(E):
        print(idE)
        H_eff = 0
        for idk, item_k in enumerate(kz):
            gl = Alpha.self_energy(energy_item, Hs[idk], Hls[idk], Ss[idk], Sls[idk])
            sigma_l = (energy_item*Sls[idk] - Hls[idk]) @ gl @ np.matrix.getH(energy_item*Sls[idk] - Hls[idk])

            gr = Alpha.self_energy(energy_item, Hs[idk], Hsr[idk], Ss[idk], Ssr[idk])
            sigma_r = (energy_item*Ssr[idk] - Hsr[idk]) @ gr @ np.matrix.getH(energy_item*Ssr[idk] - Hsr[idk])

            H22 = H_graphene[idk] + sigma_l + sigma_r
            H_eff += H_beta[idk] @ (energy_item*np.eye(H22.shape[0]) - H22) @ np.matrix.getH(H_beta[idk])

        sigma11[idE] = H_eff/NK


    sigma11_diag = np.diag(np.imag(sigma11))

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(2,2,1)
    ax1.hist(sigma11_diag[0::4])
#     ax1.ylim(0, 0.03)
    ax1.set_title('s orbital')

#     ax2 = fig2.add_subplot(2,2,2)
#     ax2.plot(sigma11_diag[1::4])
#     ax2.set_title('py orbital')
# 
#     ax3 = fig2.add_subplot(2,2,3)
#     ax3.plot(sigma11_diag[2::4])
#     ax3.set_title('pz orbital')
# 
#     ax4 = fig2.add_subplot(2,2,4)
#     ax4.plot(sigma11_diag[3::4])
#     ax4.set_title('px orbital')
    
#     fig1 = plt.figure()
#     ax5 = fig1.add_subplot(1,1,1)
#     ax5.plot(sigma11_diag)
#     plt.plot(np.diag(np.imag(sigma11)))
#     plt.xlabel("E")
#     plt.ylabel("DOS")
#     plt.grid()
    # plt.savefig('fig.png')
    plt.show()
