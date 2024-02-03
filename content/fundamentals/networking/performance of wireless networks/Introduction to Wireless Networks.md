---
title: Introduction to Wireless Networks
---
# Ubiquitous Connectivity
Nowadays, we have come to expect ubiquitous connectivity.

Wireless networks make this possible.  They are based on common proinciples, tradeoffs, and have common constraints.

# Types of Wireless Networks

Network: group of devices connected to one another

Wireless network: typically uses radio communication as medium

There are lots of technologies for lots of applications, so there are many ways to classify different wireless networks

# Performance Fundamentals of Wireless Networks

Every wireless technology has its own set of constraints, but they all have a **maximum channel capacity**, determined by the same underlying principles
- $C = BW \times \text{log}_2(1 + \frac{S}{N})$ 
	- $C$: channel capacity (bits per second)
	- $BW$: available bandwidth (hertz)
	- $S$ signal (watts)
	- $N$ noise (watts)
## Bandwidth

Radio communication uses a shared medium: radio waves!

So, sender + reciever need to agree on specific requency range over which communation will occur

Who determines the requency range and its allocation?
- In the US- the TCC

What matters for performance? he size of the asigned frequency range (channel bit rate directly proportional to this)

Note: not all frequency ranges ofer the same performance
- Low frequency: travel father and larger areas- require larger antennas
- High frequency: transfer more ata but not as far

## Signal Power
This is the second fundamental limiting power

Signal power: compares level of desired signal to level of background noise and interference
- Larger amount of background noise- stronger signal needs to be

Remember: shared medium- get unwanted interference

2 effects
- Near-far problem: when strong signal crowds out weaker signal such that receiver can’t detect it
- Cell-breathing: condition where coverage area expands and shrinks based on cumulative noise and interference levels


## Modulation
Algorithm by which signal is encoded can also have a significant effect

**Modulation**: digital (1s and 0s)-to-analog (radio waves) conversion

Example
- Receiver/sender can process 1,000 pulses/symbols per second (1,000 baud)
- Each symbol represents a different bit-sequence, (e.g. in 2-bit. alphabet)
- Bit rate of channel is 1,000 x 2 bits per symbol (2,000 bits per second)

Higher order moduilation alphabet comes at cost of reduced robustness to noise and interference


# Measuring Real-World Wireless Performance
In summary

All radio-powered communication is:
- Done over radio waves (shared medium)
- Regulated to use specific:
	- Bandwidth frequency ranges
	- Transmit power rates
- Subject to:
	- Continuously changing background noise/intererence
	- Constraints of chosen wireless tech and device

The performance of any wireless network is affected by just a few factors:
- Distance between receiver + sender
- Background noise in current location
- Interference from users in same network
- Interference from users in other, nearby networks
- Processing power + chosen modulation scheme

Basically, to get maximum throughjput
- Place receiver and sender as close as posible
- Give all the power they want
- Select best modulation mehtod

