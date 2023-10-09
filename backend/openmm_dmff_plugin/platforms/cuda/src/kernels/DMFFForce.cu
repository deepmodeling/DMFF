extern "C" __global__
void addForces(const FORCES_TYPE* __restrict__ forces, long long* __restrict__ forceBuffers, int* __restrict__ atomIndex, int numAtoms, int paddedNumAtoms) {
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
        int index = atomIndex[atom];
        forceBuffers[atom] += (long long) (forces[3*index]*0x100000000);
        forceBuffers[atom+paddedNumAtoms] += (long long) (forces[3*index+1]*0x100000000);
        forceBuffers[atom+2*paddedNumAtoms] += (long long) (forces[3*index+2]*0x100000000);
    }
}

