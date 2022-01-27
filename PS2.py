import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ButeraSmith():

    def __init__(self, tmax, el, is_reduction, is_plotting_nullcline):
        self.is_reduction = is_reduction
        self.is_plotting_nullcline = is_plotting_nullcline
        self.tmax = tmax

        """Full Butera-Smith Model implemented in Python"""

        self.cm = 21.0
        """membrane capacitance, in uF/cm^2"""

        self.gna = 28.0
        if self.is_reduction: self.gna = 0
        """Sodium (Na) maximum conductance, in mS/cm^2"""

        self.gk  =  11.2
        """Postassium (K) maximum conductance, in mS/cm^2"""

        self.gl = 2.8
        """Leak maximum conductance, in mS/cm^2"""

        self.ena =  50.0
        """Sodium (Na) Nernst reversal potential, in mV"""

        self.ek  = -85.0
        """Postassium (K) Nernst reversal potential, in mV"""

        self.el  = el
        """Leak Nernst reversal potential, in mV"""

        self.t = np.arange(0.0, tmax, 0.01)
        """ The time to integrate over """

        self.i = 0.0
        """ Applied current """

        self.taubar = 10000.0

        self.gnap = 2.8

    def xinf(self, v, vt, sig):
        return 1/(1+np.exp((v-vt)/sig))
    
    def taux(self, v, vt, sig, tau):
        return tau/np.cosh((v-vt)/(2*sig))
    
    def il(self, v):
        return self.gl * (v - self.el)
    
    def minf(self, v):
        return self.xinf(v, -34.0, -5.0)

    def ina(self, v, n):
        return self.gna * (self.minf(v) ** 3) * (1-n) * (v-self.ena)
    
    def ninf(self, v):
        return self.xinf(v, -29.0, -4.0)

    def taun(self, v):
        return self.taux(v, -29.0, -4.0, 10.0)

    def ik(self, v, n):
        return self.gk * (n**4) * (v-self.ek)
    
    def mninf(self, v):
        return self.xinf(v, -40.0, -6.0)

    def hinf(self, v):
        return self.xinf(v, -48.0, 6.0)

    def tauh(self, v):
        return self.taux(v, -48.0, 60.0, self.taubar)
    
    def inap(self, v, h):
        return self.gnap * self.mninf(v) * h * (v-self.ena)

    def v_nullcline(self, v):
        n = self.ninf(v)
        return (self.i - self.il(v) - self.ina(v, n) - self.ik(v, n)) / (self.gnap * self.mninf(v) * (v-self.ena))
    
    def h_nullcline(self, v):
        return 1 / (1 + np.exp((v + 48.0) / 12))

    def graph_nullcline(self, v_range):
        v_nullcline_y = []
        h_nullcline_y = []

        for v in np.nditer(v_range):
            v_nullcline_y.append(self.v_nullcline(v))
            h_nullcline_y.append(self.h_nullcline(v))

        return v_nullcline_y, h_nullcline_y

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        v, n, h = X
        if self.is_reduction: n = self.ninf(v)

        dvdt = (self.i - self.il(v) - self.ina(v, n) - self.ik(v, n) - self.inap(v, h)) / self.cm
        dndt = (self.ninf(v) - n) / self.taun(v)
        dhdt = (self.hinf(v) - h) / self.tauh(v)

        return dvdt, dndt, dhdt

    def Main(self):
        """
        Main demo for the Butera Smith neuron model
        """

        X = odeint(self.dALLdt, [-55.0, 0.0, 0.6], self.t, args=(self,))
        V = X[:,0]
        n = X[:,1]
        h = X[:,2]

        v_range = np.arange(-62, -25, .1)
        v_nullcline_y, h_nullcline_y = self.graph_nullcline(v_range)

        ina = self.ina(V, n)
        ik = self.ik(V, n)
        il = self.il(V)

        if not self.is_plotting_nullcline:
            num_plots = 3
            fig, axs = plt.subplots(num_plots)
            fig.suptitle('Butera-Smith')
            fig.subplots_adjust(hspace=.5)
            axs[0].plot(self.t, V, 'k')
            axs[0].set(ylabel='V (mV)')
            axs[1].plot(self.t, ina, 'c', label='$I_{Na}$')
            axs[1].plot(self.t, ik, 'y', label='$I_{K}$')
            axs[1].plot(self.t, il, 'm', label='$I_{L}$')
            axs[1].set(ylabel='Currents')
            axs[1].legend()
            # axs[2].plot(self.t, m, 'r', label='m')
            axs[2].plot(self.t, h, 'g', label='h')
            axs[2].plot(self.t, n, 'b', label='n')
            axs[2].set(ylabel='Gating Variables')
            axs[2].legend()
            # i_inj_values = [self.I_inj(t) for t in self.t]
            # axs[3].plot(self.t, i_inj_values, 'k')
            # axs[3].set(xlabel='t (ms)',ylabel='$I_{inj}$ ($\\mu{A}/cm^2$)')
            # axs[3].plot(V, h)
            # axs[3].set(xlabel='V (mv)', ylabel='h')

            # axs[4].plot(list(v_range), v_nullcline_y)
            # axs[4].set(xlabel='V (-100 to 30 mv)', ylabel='h (V nullcline)')

            # axs[4].plot(list(v_range), h_nullcline_y)
            # axs[4].set(xlabel='V (-100 to 30 mv)', ylabel='h (h nullcline)')

            plt.savefig('BS-I10.jpg', format = 'jpg')

            plt.show()
        else:
            line1, = plt.plot(V,h, label='$Phase Plane$')
            line2, = plt.plot(list(v_range), v_nullcline_y)
            line3, = plt.plot(list(v_range), h_nullcline_y)
            plt.xlabel("V")
            plt.ylabel("h")
            plt.legend([line1, line2, line3], ['Phase Plane', 'V Nullcline', 'H Nullcline'])
            plt.show()


if __name__ == '__main__':
    # runner = ButeraSmith(tmax=1000.0, el=-65.0, is_reduction=True, is_plotting_nullcline=False)
    # runner.Main()
    runner = ButeraSmith(tmax=50000.0, el=-60.0, is_reduction=True, is_plotting_nullcline=True)
    runner.Main()


"""
# butera and smith model using NaP
par cm=21,i=0
xinf(v,vt,sig)=1/(1+exp((v-vt)/sig))
taux(v,vt,sig,tau)=tau/cosh((v-vt)/(2*sig))

# leak
il=gl*(v-el)
par gl=2.8,el=-65

# fast sodium --  h=1-n
minf(v)=xinf(v,-34,-5)
ina=gna*minf(v)^3*(1-n)*(v-ena)
par gna=28,ena=50

# delayed rectifier
ninf(v)=xinf(v,-29,-4)
taun(v)=taux(v,-29,-4,10)
ik=gk*n^4*(v-ek)
par gk=11.2,ek=-85

# NaP
mninf(v)=xinf(v,-40,-6)
hinf(v)=xinf(v,-48,6)
tauh(v)=taux(v,-48,6,taubar)
par gnap=2.8,taubar=10000
inap=gnap*mninf(v)*h*(v-ena)

v' = (i-il-ina-ik-inap)/cm
n'=(ninf(v)-n)/taun(v)
h'=(hinf(v)-h)/tauh(v)

"""