import numpy as np
from numpy import pi, sin, cos, log10, exp, zeros, mod
import matplotlib.pyplot as plt
import plldesigner.pll as pll

class Sim(object):
    def __init__(self, resultdir, ts, tend):
        self.resuldir = resultdir
        self.ts = ts
        self.tend = tend 
        self.n = int(tend/ts)
        
class Struct(object):
    pass
Icp = 300e-6;
mypll=pll.AnalogPLL(3, 5.218e+08, Navg=55.22, prescaler=2, 
                    plltype='fractionalN')
mypll.loopcalc(1e6, 60.0, -107.8, 1e6, 0.7, 300)

if __name__ == "__main__":
    sim_tran = Sim('./sim/', 1e-9, 500e-6)
    phivco = zeros(sim_tran.n+1) + np.random.randn(1)
    phidiv = zeros(sim_tran.n+1)
    phiref = zeros(sim_tran.n+1)
    vcont =  zeros(sim_tran.n+1)
    icp =  zeros(sim_tran.n+1)
    fo =  zeros(sim_tran.n+1)
    fref = 480e6
    vco = pll.vco(300e6, 26e9, 0.5)
    adpll = Struct()
    adpll.N = 57.23
    
    for i in np.arange(sim_tran.n)+1:
        fo[i] = vco.freq_vs_volt(vcont[i])
        phivco[i] = phivco[i-1]+2*pi*fo[i]*sim_tran.ts
        phidiv[i] = mod(phivco[i]/adpll.N,2*pi)
        phiref[i] = mod(phiref[i-1]+fref*sim_tran.ts,2*pi)
        icp[i] = Icp/(phiref[i]-phidiv[i])/2/pi
        
    # Remco tesis 
    # NsigmaNT = sim.Ts*VCO.Ffr;
    #sigmaNT  = sqrt(NsigmaNT * 10^(VCO.Lvco1M/10) * 1e6^2 / VCO.Ffr^3);
    # periodnoise = sigmaNT*randn(1,sim.n+1);
 