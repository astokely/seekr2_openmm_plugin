"""
This demonstration runs the trypsin/benzamidine system as an example for the
plugin.
"""

import sys
import os
from time import time

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from seekr2plugin import MmvtLangevinIntegrator, vectori, vectord
import seekr2plugin

lig_indices = [3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229]
rec_indices = [2478, 2489, 2499, 2535, 2718, 2745, 2769, 2787, 2794, 2867, 2926]

ben_central_alpha_carbon = [3221]

angle_cv1_atom1 = [2799]
angle_cv1_atom2 = [2794]

angle_cv2_atom1 = [2512]
angle_cv2_atom2 = [1858]

angle_cv3_atom1 = [2936]
angle_cv3_atom2 = [2541]

angle_cv4_atom1 = [2494]
angle_cv4_atom2 = [2491]

angle_cv5_atom1 = [3225]
angle_cv5_atom2 = [3224]

box_vector = Quantity(
    [[61.23940982865766, 0.0, 0.0], 
     [-20.413134962473007, 57.73706986993814, 0.0], 
     [-20.413134962473007, -28.868531440986094, 50.00177126469543]], 
                      unit=angstrom)

prmtop = AmberPrmtopFile('tryp_ben.prmtop')
inpcrd = AmberInpcrdFile('tryp_ben.inpcrd')

system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

mypdb = PDBFile('tryp_ben_new.pdb')


cvforce = CustomCentroidBondForce(
    13,
    "k1*step(-1*(distance(g1, g2)^2-radius1^2))"
    + "+ k2*step(distance(g1, g2)^2 - radius2^2)"
    + "+ k3*step(-1*(angle(g3, g4, g5) - angle1))"
    + "+ k4*step(angle(g3, g4, g5) - angle2)"
    + "+ k5*step(-1*(angle(g3, g6, g7) - angle3))"
    + "+ k6*step(angle(g3, g6, g7) - angle4)"
    + "+ k7*step(-1*(angle(g3, g8, g9) - angle5))"
    + "+ k8*step(angle(g3, g8, g9) - angle6)"
    + "+ k9*step(-1*(angle(g3, g10, g11) - angle7))"
    + "+ k10*step(angle(g3, g10, g11) - angle8)"
    + "+ k11*step(-1*(angle(g3, g10, g11) - angle9))"
    + "+ k12*step(angle(g3, g12, g13) - angle10)"
)
rec_group = cvforce.addGroup(rec_indices)
lig_group = cvforce.addGroup(lig_indices)

ben_central_alpha_carbon_group = cvforce.addGroup(
    ben_central_alpha_carbon
)

angle_cv1_atom1_group = cvforce.addGroup(angle_cv1_atom1)
angle_cv1_atom2_group = cvforce.addGroup(angle_cv1_atom2)

angle_cv2_atom1_group = cvforce.addGroup(angle_cv2_atom1)
angle_cv2_atom2_group = cvforce.addGroup(angle_cv2_atom2)

angle_cv3_atom1_group = cvforce.addGroup(angle_cv3_atom1)
angle_cv3_atom2_group = cvforce.addGroup(angle_cv3_atom2)

angle_cv4_atom1_group = cvforce.addGroup(angle_cv4_atom1)
angle_cv4_atom2_group = cvforce.addGroup(angle_cv4_atom2)

angle_cv5_atom1_group = cvforce.addGroup(angle_cv5_atom1)
angle_cv5_atom2_group = cvforce.addGroup(angle_cv5_atom2)

cvforce.setForceGroup(1)
cvforce.addPerBondParameter('k1')
cvforce.addPerBondParameter('k2')
cvforce.addPerBondParameter('k3')
cvforce.addPerBondParameter('k4')
cvforce.addPerBondParameter('k5')
cvforce.addPerBondParameter('k6')
cvforce.addPerBondParameter('k7')
cvforce.addPerBondParameter('k8')
cvforce.addPerBondParameter('k9')
cvforce.addPerBondParameter('k10')
cvforce.addPerBondParameter('k11')
cvforce.addPerBondParameter('k12')
cvforce.addPerBondParameter('radius1')
cvforce.addPerBondParameter('radius2')
cvforce.addPerBondParameter('angle1')
cvforce.addPerBondParameter('angle2')
cvforce.addPerBondParameter('angle3')
cvforce.addPerBondParameter('angle4')
cvforce.addPerBondParameter('angle5')
cvforce.addPerBondParameter('angle6')
cvforce.addPerBondParameter('angle7')
cvforce.addPerBondParameter('angle8')
cvforce.addPerBondParameter('angle9')
cvforce.addPerBondParameter('angle10')
cvforce.addBond(
    [
        rec_group, lig_group,
        ben_central_alpha_carbon_group,
        angle_cv1_atom1_group,
        angle_cv1_atom2_group,
        angle_cv2_atom1_group,
        angle_cv2_atom2_group,
        angle_cv3_atom1_group,
        angle_cv3_atom2_group,
        angle_cv4_atom1_group,
        angle_cv4_atom2_group,
        angle_cv5_atom1_group,
        angle_cv5_atom2_group,
    ], 
    [
        1, 2, 4, 8, 16, 32, 64, 
        128, 256, 512, 1024, 2048, 
        11.0*angstroms, 13.0*angstroms, 
        130.0*degrees, 145.0*degrees,
        25.0*degrees, 40.0*degrees,
        20.0*degrees, 35.0*degrees,
        130.0*degrees, 145.0*degrees,
        80.0*degrees, 95.0*degrees,
    ]
)
CvForce = system.addForce(cvforce)

integrator = MmvtLangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds, "6_cv.txt")
integrator.addMilestoneGroup(1)

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaDeviceIndex': '0', 'CudaPrecision': 'mixed'}
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(mypdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)

simulation.context.setPeriodicBoxVectors(*box_vector)
    
simulation.reporters.append(PDBReporter('6_cv_output.pdb', 1000))
simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True))
num_steps = 100000
start = time()
simulation.step(num_steps)
print((num_steps * 2 * 1e-6) / ((time() - start) / (3600 * 24)))

