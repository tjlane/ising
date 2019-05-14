
"""
A simple N-dimensional Ising Monte-Carlo model
"""

import tables
import numpy as np
from scipy import signal


def gaussian_beam_kernel(shape, sigma):
    N = len(shape)
    k = np.ones(shape)
    for a in range(N):
        s = [None,] * N
        s[a] = slice(None,None)
        k *= signal.gaussian(shape[a], std=sigma)[tuple(s)]
    return k


def fft_convolve(a, b):
    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    c = np.abs(np.fft.ifftn(fa * fb))
    return np.fft.fftshift(c)


def lattice_spatial_correlation(lattices_a, lattices_b=None):
    """
    Compute the spatial correlation for a series of lattices.
    
    Parameters
    ----------
    lattices : np.ndarray
        The first dimension indexes the lattice, the last N-1 dimensions
        should be spatial.
        
    Returns
    -------
    correlation : np.ndarray
        An array of the same shape as `lattices` with the spatial
        correlation of each lattice.
    """

    axes = range(1, len(lattices.shape))
    Fa = np.fft.fftn(lattices_a, axes=axes)

    if lattices_b is None:
        Fb = Fa
    else:
        Fb = np.fft.fftn(lattices_b, axes=axes)

    iF2 = np.fft.ifftn(Fa * np.conj(Fb), axes=axes)
    c = np.abs(np.fft.fftshift(iF2, axes=axes))
    c /= c.max()

    return c


def scatter_lattice(lattice, de=0.5, beam_kernel=None):
    
    x = lattice.copy().astype(np.float) # do not modify
    x[x == -1] = 1.0 - de
    
    if beam_kernel is not None:
        if not (beam_kernel.shape == lattice.shape):
            raise ValueError('beam_kernel and lattice must have same shape')
        x *= beam_kernel
    
    intensity = np.square(np.abs(np.fft.fftshift(np.fft.fftn(x))))
    
    return intensity


def radial_average(image, n_bins=101):

    mg_arg = [np.linspace(-x/2., x/2., x) for x in space_acf.shape]
    mg = np.meshgrid(*mg_arg)
    r = np.sqrt(np.sum(np.square(mg), axis=0))

    y, x = np.histogram(r, bins=np.linspace(0.0, r.max(), n_bins+1), 
                        weights=space_acf * np.power(r,-len(space_acf.shape)))

    return x[:-1], y


class Model(object):
    """
    ising model in the absence of a field
    >> any dimension
    >> periodic boundaries
    >> metropolis propogation

    positive bj: spins want to align, ferromagnet
    """


    def __init__(self, shape, bj, save_interval=None, verbose=False):

        self.shape = shape
        self.lattice = np.random.binomial(1, 0.5, shape)
        self.lattice[ self.lattice == 0 ] = -1

        self.bj = bj

        self.energies = []
        self.mags     = []

        self.accepted = []

        self.verbose = verbose

        self.save_interval = save_interval
        if save_interval:
            shpstr = '-'.join(['%s' % x for x in self.shape])
            self.file_name = 'ising_bj%f_shape%s.h5' % (bj, shpstr)
            self.file = tables.File(self.file_name, 'w')
            a = tables.Atom.from_dtype(np.dtype(np.float64))
            self._saved_lattices = self.file.create_earray(where='/', 
                                               name='lattices',
                                               shape=tuple([0] + list(self.shape)), 
                                               atom=a)
        return


    @property
    def dimension(self):
        return len(self.shape)


    @property
    def energy(self):

        e = 0.0

        for a in range(self.dimension):
            e_x1 = - 2.0 * np.sum(self.lattice * np.roll(self.lattice,  1, axis=a))
            e += self.bj * e_x1

        return e


    @property
    def mag_ac(self):
        ac = np.correlate(self.mags, self.mags, 'full')[len(self.mags):]
        ac /= ac[0]
        return ac


    def mc_steps(self, n_steps, flips_per_move=1):

        for i in range(n_steps):

            current = self.energy
            old_lattice = self.lattice.copy()

            for f in range(flips_per_move):
                idx = []
                for a in range(self.dimension):
                    idx.append( np.random.randint(self.shape[a]) )
                self.lattice[tuple(idx)] *= -1 # flip spin

            new = self.energy
            p = min(1, np.exp(-1.0 * (new-current)))
            if self.verbose:
                print('MC step: %.2f %.2f | %.2f' % (current, new, p))

            v = np.random.random()
            if v < p:
                if self.verbose: print('\t accepted')
                self.accepted.append(1)
            else:
                if self.verbose: print('\t rejected')
                self.lattice = old_lattice
                self.accepted.append(0)

            self.energies.append(self.energy)
            self.mags.append(np.sum(self.lattice))

            if self.save_interval is not None:
                if (i % self.save_interval == 0):
                    print('move %d, saving lattice' % i)
                    self._saved_lattices.append(self.lattice[None,...])
                    print(self.lattice[None,...].shape)

        if self.save_interval is not None:
            # save other info
            float_a = tables.Atom.from_dtype(np.dtype(np.float64))
            int_a   = tables.Atom.from_dtype(np.dtype(np.int64))

            e = self.file.create_carray(self.file.root, 'energies', 
                                        float_a, shape=(len(self.energies),))
            e[:] = np.array(self.energies)
            m = self.file.create_carray(self.file.root, 'mags', float_a, shape=(len(self.mags),))
            m[:] = np.array(self.mags)
            a = self.file.create_carray(self.file.root, 'accepted', int_a, shape=(len(self.accepted),))
            a[:] = np.array(self.accepted)

        return


if __name__ == '__main__':
    
    m = Model((16,16), 4.0, 100.0)
    print(m.energy)
    m.mc_steps(10)
    print(m.energies)
    print(m.mags)


