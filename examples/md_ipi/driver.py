"""
---------------------------------------------------------------------
|I-PI socket client.
|
|Version: 0.1
|Program Language: Python 3.6
|Developer: Xinyan Wang
|Homepage:https://github.com/WangXinyan940/i-pi-driver
|
|Receive coordinate and send force back to i-PI server using socket.
|Read http://ipi-code.org/assets/pdf/manual.pdf for details.
---------------------------------------------------------------------
"""
import os
import socket
import struct
import numpy as np
import sys

# CONSTANTS
BOHR = 5.291772108e-11  # Bohr -> m
ANGSTROM = 1e-10  # angstrom -> m
AMU = 1.660539040e-27  # amu -> kg
FEMTO = 1e-15
PICO = 1e-12
EH = 4.35974417e-18  # Hartrees -> J
EV = 1.6021766209e-19  # eV -> J
H = 6.626069934e-34  # Planck const
KB = 1.38064852e-23  # Boltzmann const
MOLE = 6.02214129e23
KJ = 1000.0
KCAL = 4184.0
# HEADERS
STATUS = b"STATUS      "
NEEDINIT = b"NEEDINIT    "
READY = b"READY       "
HAVEDATA = b"HAVEDATA    "
FORCEREADY = b"FORCEREADY  "
# BYTES
INT = 4
FLOAT = 8


class ExitSignal(BaseException):
    pass


class TimeOutSignal(BaseException):
    pass

class BaseDriver(object):
    """
    Base class of Socket driver.
    """

    def __init__(self, port, addr="127.0.0.1", socket_type='inet'):
        if socket_type == 'inet':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif socket_type == 'unix':
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        else:
            sys.exit('Unknown socket type:', socket_type)
        self.socket_type = socket_type
        self.socket.settimeout(1000)
        try:
            if socket_type == 'inet':
                self.socket.connect((addr, port))
            elif socket_type == 'unix':
                print('/tmp/ipi_'+addr)
                self.socket.connect('/tmp/ipi_'+addr)
            else:
                sys.exit('Unknown socket type:', socket_type)
            self.socket.settimeout(None)
        except socket.timeout as e:
            raise TimeOutSignal("Time out, quit.")
        self.ifInit = False
        self.ifForce = False
        self.cell = None
        self.inverse = None
        self.crd = None
        self.energy = None
        self.force = None
        self.extra = b""
        self.nbead = -1
        self.natom = -1

    # cell interface added by Kuang
    def grad(self, crd, cell):
        """
        Calculate gradient.
        Need to be rewritten in inheritance.
        """
        return None, None

    def update(self, text):
        """
        Update system message from INIT motion.
        Need to be rewritten in inheritance.
        Mostly we don't need it.
        """
        pass

    def init(self):
        """
        Deal with message from INIT motion.
        """
        self.nbead = np.frombuffer(
            self.socket.recv(INT * 1), dtype=np.int32)[0]
        offset = np.frombuffer(self.socket.recv(INT * 1), dtype=np.int32)[0]
        self.update(self.socket.recv(offset))
        self.ifInit = True

    def status(self):
        """
        Reply STATUS.
        """
        if self.ifInit and not self.ifForce:
            self.socket.send(READY)
        elif self.ifForce:
            self.socket.send(HAVEDATA)
        else:
            self.socket.send(NEEDINIT)

    def posdata(self):
        """
        Read position data.
        """
        self.cell = np.frombuffer(self.socket.recv(
            FLOAT * 9), dtype=np.float64) * BOHR
        self.inverse = np.frombuffer(self.socket.recv(
            FLOAT * 9), dtype=np.float64) / BOHR
        self.natom = np.frombuffer(
            self.socket.recv(INT * 1), dtype=np.int32)[0]
        if (self.socket_type == 'unix'):
            crd = np.frombuffer(self.socket.recv(FLOAT * 3 * self.natom, socket.MSG_WAITALL), dtype=np.float64)
        else:
            crd = np.frombuffer(self.socket.recv(FLOAT * 3 * self.natom), dtype=np.float64)
        self.crd = crd.reshape((self.natom, 3)) * BOHR
        # added by Kuang
        self.cell = self.cell.reshape((3, 3)).T
        self.inverse = self.inverse.reshape((3, 3)).T
        energy, force = self.grad(self.crd, self.cell)
        self.energy = energy
        self.force = - force
        self.ifForce = True

    def getforce(self):
        """
        Reply GETFORCE.
        """
        self.socket.send(FORCEREADY)
        self.socket.send(struct.pack("d", self.energy / EH))
        self.socket.send(struct.pack("i", self.natom))
        for f in self.force.ravel():
            self.socket.send(struct.pack("d", f / (EH / BOHR))
                             )  # Force unit: xx
        virial = np.diag((self.force * self.crd).sum(axis=0)).ravel() / EH
        for v in virial:
            self.socket.send(struct.pack("d", v))
        extra = self.extra if len(self.extra) > 0 else b" "
        lextra = len(extra)
        self.socket.send(struct.pack("i", lextra))
        self.socket.send(extra)
        self.ifForce = False

    def exit(self):
        """
        Exit.
        """
        self.socket.close()
        raise ExitSignal()

    def parse(self):
        """
        Reply the request from server.
        """
        try:
            self.socket.settimeout(1000)
            header = self.socket.recv(12).strip()
            self.socket.settimeout(None)
        except socket.timeout as e:
            raise TimeOutSignal("Time out, quit.")
        if len(header) < 2:
            raise TimeOutSignal()
        if header == b"STATUS":
            self.status()
        elif header == b"INIT":
            self.init()
        elif header == b"POSDATA":
            self.posdata()
        elif header == b"GETFORCE":
            self.getforce()
        elif header == b"EXIT":
            self.exit()


class HarmonicDriver(BaseDriver):
    """
    Driver for ideal gas molecule with harmonic potential.
    Just for test.
    """

    def __init__(self, port, addr, k):
        BaseDriver.__init__(self, port, addr, 'inet')
        self.kconst = k * (KJ / MOLE)

    def grad(self, crd, cell):
        r = (crd ** 2).sum(axis=1)
        energy = (self.kconst * r ** 2).sum()
        grad = 2 * self.kconst * crd / r.reshape((-1, 1))
        return energy, grad


class GaussDriver(BaseDriver):
    """
    Driver for QM calculation with Gaussian.
    """

    def __init__(self, port, addr, template, atoms, path="g09"):
        BaseDriver.__init__(self, port, addr, 'inet')
        with open(template, "r") as f:
            text = f.readlines()
        self.template = text
        self.atoms = atoms
        self.gau = path

    def gengjf(self, crd):
        """
        Generate .gjf file.
        """
        with open("tmp.gjf", "w") as f:
            for line in self.template:
                if "[coord]" in line:
                    for i in range(len(self.atoms)):
                        f.write("%s %16.8f %16.8f %16.8f\n" %
                                (self.atoms[i], crd[i, 0], crd[i, 1], crd[i, 2]))
                else:
                    f.write(line)

    def readlog(self):
        """
        Get energy and force from .log file.
        """
        with open("tmp.log", "r") as f:
            text = f.readlines()
        natoms = len(self.atoms)
        ener = [i for i in text if "SCF Done:" in i]
        if len(ener) != 0:
            ener = ener[-1]
            ener = np.float64(ener.split()[4])
        else:
            ener = np.float64(
                [i for i in text if "Energy=" in i][-1].split()[1])
        for ni, li in enumerate(text):
            if "Forces (Hartrees/Bohr)" in li:
                break
        forces = text[ni + 3:ni + 3 + natoms]
        forces = [i.strip().split()[-3:] for i in forces]
        forces = [[np.float64(i[0]), np.float64(i[1]), np.float64(i[2])]
                  for i in forces]
        return ener, - np.array(forces)

    def grad(self, crd, cell):
        self.gengjf(crd / ANGSTROM)
        os.system("%s tmp.gjf" % self.gau)
        energy, grad = self.readlog()
        energy = energy * EH
        grad = grad * (EH / BOHR)
        return energy, grad


if __name__ == '__main__':
    driver = HarmonicDriver(31415, "127.0.0.1", 100.0)
    while True:
        driver.parse()
