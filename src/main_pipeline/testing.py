from pulser import Register, Sequence, Pulse
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import BlackmanWaveform

reg = Register({
  "target": (-2, 0),
  "control": (2, 0),
})

seq = Sequence(reg, DigitalAnalogDevice)
seq.declare_channel("digital", "raman_local")
seq.declare_channel("rydberg", "rydberg_local")

amp1 = BlackmanWaveform(200, 1.5707963267948966)
pulse1 = Pulse.ConstantDetuning(amp1, 0, 4.71238898038469)

amp2 = BlackmanWaveform(200, 1.5707963267948966)
pulse2 = Pulse.ConstantDetuning(amp2, 0, 1.5707963267948966)

amp3 = BlackmanWaveform(200, 3.141592653589793)
pulse3 = Pulse.ConstantDetuning(amp3, 0, 0)

amp4 = BlackmanWaveform(725, 6.283185307179586)
pulse4 = Pulse.ConstantDetuning(amp4, 0, 0)

amp5 = BlackmanWaveform(200, 3.141592653589793)
pulse5 = Pulse.ConstantDetuning(amp5, 0, 0)

amp6 = BlackmanWaveform(200, 1.5707963267948966)
pulse6 = Pulse.ConstantDetuning(amp6, 0, 4.71238898038469)

seq.target("target", "digital")
seq.add(pulse1, "digital")
seq.target("control", "digital")
seq.add(pulse2, "digital")
seq.target("target", "rydberg")
seq.align("digital", "rydberg")
seq.add(pulse3, "rydberg")
seq.target("control", "rydberg")
seq.add(pulse4, "rydberg")
seq.target("target", "rydberg")
seq.add(pulse5, "rydberg")
seq.align("rydberg", "digital")
seq.add(pulse6, "digital")
seq.measure("digital")

seq.draw()