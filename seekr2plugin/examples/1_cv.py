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
    2, 
    "k1*step(-1*(distance(g1,g2)^2-radius1^2))"
    + "+ k2*step(distance(g1, g2)^2 - radius2^2)"
)
rec_group = cvforce.addGroup(rec_indices)
lig_group = cvforce.addGroup(lig_indices)
cvforce.setForceGroup(1)
cvforce.addPerBondParameter('k1')
cvforce.addPerBondParameter('k2')
cvforce.addPerBondParameter('radius1')
cvforce.addPerBondParameter('radius2')
cvforce.addBond([rec_group, lig_group], [1, 2, 11.0*angstroms, 13.0*angstroms])
CvForce = system.addForce(cvforce)

integrator = MmvtLangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds, "1_cv.txt")
integrator.addMilestoneGroup(1)

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaDeviceIndex': '0', 'CudaPrecision': 'mixed'}
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(mypdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)

simulation.context.setPeriodicBoxVectors(*box_vector)
    
#simulation.reporters.append(PDBReporter('tryp_ben_output.pdb', 1000))
simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True))
simulation.step(25000)

