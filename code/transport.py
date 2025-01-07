## Mathematical Modeling in GAM
##
## The module is designed to calculate the energy characteristics of 
## atomic and molecular species using nitrogen as an example.
##
## 2024-2025, Batalov S.A.

# Importing required modules
import numpy as np
import math as math

# Conversion of atomic mass units into kilograms
atm2Kg = 1.66e-27

# Boltzmann's constant
k = 1.380658e-23

# Planck's constant
h = 6.62607015e-34

# Speed of light
c = 2.99792458e8

# Conversion of atomic mass units into kilograms
def AtomicToKg(atomicMass):
    return atomicMass * atm2Kg

# Translating tabular data into SI
def EnergyToSi(energy):
    return np.array(energy) * 1e2 * h * c

# A class containing tools for calculating statsums, 
# specific energies and heats of monoatomic chemical sort
class Atom:
    
    # Initialization with file
    def __init__(self, atomicMass, fileName):
        inputMat  = np.loadtxt(fileName)
        self.mass = AtomicToKg(atomicMass)
        self.gN   = inputMat[:, 0]
        self.epsN = EnergyToSi(inputMat[:, 1])

    # Compute internal energy of electronic excitation
    def EpsElN(self, n):
        return self.epsN[n]
    
    # Compute full internal energy
    def Eps(self, n):
        return self.EpsElN(n)

    # Compute electronic excitation statweights
    def GN(self, n):
        return self.gN[n]
    
    # Compute translational statsum
    def StatSumTr(self, t):
        return (math.tau * self.mass * k * t / h ** 2) ** 1.5

    # Compute internal electronic excitation statsum
    def StatSumInt(self, t):
        temp = 0.0
        for n in range(self.gN.size):
            temp += self.GN(n) * math.exp(-self.Eps(n) / k / t)
        return temp
    
    # Compute full statsum
    def StatSumFull(self, t):
        return self.StatSumTr(t) * self.StatSumInt(t)

    # Compute translational specific energy
    def EnergyTr(self, t):
        return 1.5 * k * t / self.mass

    # Compute internal electronic excitation specific energy
    def EnergyInt(self, t):
        temp = 0.0
        for n in range(self.gN.size):
            temp += self.GN(n) * self.Eps(n) * math.exp(-self.Eps(n) / k / t)
        return temp / self.StatSumInt(t) / self.mass
    
    # Compute full specific energy
    def EnergyFull(self, t):
        return self.EnergyTr(t) + self.EnergyInt(t)

    # Compute translational specific heat
    def SpecificHeatTr(self, t):
        return 1.5 * k / self.mass

    # Compute specific heat of internal electronic excitation
    def SpecificHeatInt(self, t):
        a = b = 0.0;
        for n in range(self.gN.size):
            a += (self.Eps(n) / k / t) ** 2 * self.GN(n) * math.exp(-self.Eps(n) / k / t)
            b += self.Eps(n) / k / t * self.GN(n) * math.exp(-self.Eps(n) / k / t)
        a /= self.StatSumInt(t);
        b /= self.StatSumInt(t);
        return k / self.mass * (a - b ** 2)

    # Compute full specific heat
    def SpecificHeatFull(self, t):
        return self.SpecificHeatTr() + self.SpecificHeatInt(t)

# A class containing tools for calculating statsums, 
# specific energies and heats of polyatomic chemical sort
class Molecule(Atom):

    # Max vibrational and rotational level numbers
    maxVibrNum = 50
    maxRotNum = 50
    
    # Initialization with file
    def __init__(self, atomicMass, fileName):
        super().__init__(atomicMass, fileName)
        inputMat   = np.loadtxt(fileName)
        self.oe    = EnergyToSi(inputMat[:, 2])
        self.oex   = EnergyToSi(inputMat[:, 3])
        self.oey   = EnergyToSi(inputMat[:, 4])
        self.oez   = EnergyToSi(inputMat[:, 5]) * 1e-3
        self.be    = EnergyToSi(inputMat[:, 6])
        self.de    = EnergyToSi(inputMat[:, 7]) * 1e-6
        self.alpha = EnergyToSi(inputMat[:, 8]) * 1e-2
        self.beta  = EnergyToSi(inputMat[:, 9])
        self.epsDiss = EnergyToSi(inputMat[:, 10])

    # Compute vibrational internal energy
    def EpsVibrNI(self, n, i):
        t = (i + 0.5)
        return self.oe[n] * t - self.oex[n] * t ** 2 + self.oey[n] * t ** 3 + self.oez[n] * t ** 4
    
    # Compute rotational internal energy
    def EpsRotNIJ(self, n, i, j):
        b = self.be[n] - self.alpha[n] * (i + 0.5)
        d = self.de[n] - self.beta[n] * (i + 0.5)
        return j * (j + 1) * (b - d * j * (j + 1))

    # Compute full internal energy
    def Eps(self, n, i, j):
        return self.EpsElN(n) + self.EpsVibrNI(n, i) + self.EpsRotNIJ(n, i, j)
    
    # Compute energy of dissociation
    def EpsDiss(self, n):
        return self.epsDiss[n]
        
    # Compute vibrational statweight
    def GI(self, i):
        return 1.0

    # Compute rotational statweight
    def GJ(self, j):
        return 2.0 * j + 1.0

    # Compute full internal statweight
    def G(self, n, i, j):
        return self.GN(n) * self.GI(i) * self.GJ(j)

    # Compute internal statsum
    def StatSumInt(self, t):
        temp = 0.0
        for n in range(self.gN.size):
            for i in range(Molecule.maxVibrNum):
                for j in range(Molecule.maxRotNum):
                    eps = self.Eps(n, i, j)
                    if eps < self.EpsDiss(n):
                        temp += self.G(n, i, j) * math.exp(-eps / k / t)
        return temp
    
    # Compute internal specific energy
    def EnergyInt(self, t):
        temp = 0.0
        for n in range(self.gN.size):
            for i in range(Molecule.maxVibrNum):
                for j in range(Molecule.maxRotNum):
                    eps = self.Eps(n, i, j)
                    if eps < self.EpsDiss(n):
                        temp += self.G(n, i, j) * eps * math.exp(-eps / k / t)
        return temp / self.StatSumInt(t) / self.mass
    
    # Compute internal specific heat
    def SpecificHeatInt(self, t):
        a = b = 0.0;
        for n in range(self.gN.size):
            for i in range(Molecule.maxVibrNum):
                for j in range(Molecule.maxRotNum):
                    eps = self.Eps(n, i, j)
                    if eps < self.EpsDiss(n):
                        a += (eps / k / t) ** 2 * self.G(n, i, j) * math.exp(-eps / k / t)
                        b += eps / k / t * self.G(n, i, j) * math.exp(-eps / k / t)
        a /= self.StatSumInt(t);
        b /= self.StatSumInt(t);
        return k / self.mass * (a - b ** 2)
        