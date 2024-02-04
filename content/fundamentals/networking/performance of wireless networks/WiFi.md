---
title: WiFi
---
Features
- Operates in unlicensed ISM spectrum
- Trivial for anywhere to deploy anywhere
- Required hardware = simple and cheap

WiFi refers to any product based on the IEEE 802.11 standards
# From Ethernet to a Wireless LAN
802.11 standards were primarily an extension of the existing Ethernet (802.3) standard
- Ethernet refered to as LAN standard
- 802.11 refered to as wireless LAN (WLAN)

All treat the shared medium as a random access channel (no central process or scheduler)

Ethernet model (CSMA) “listen before you speak”
- Medum: physical wire
- Check if anyone is transmitting
- If channel = busy, listen until free
- When channel is free, transmit immediately

WFi Model- cannot detect collisions while sending data, hence it relies on collision avoidance
- Medium: shared radio channel
- Sender transmits only when channel is sensed to be idle
- Then, sends full message frame in entirety
- Once WiFi frame sent, wats for explicit acknowledgement before proceeding with transmission

# WiFi Standards and Features
Lots of new protocols with new features

Called- 802.11b, 802.11ac, etc.

Most common today: b and g standards

How are future WiFi networks optimized? 
- Increasing bandwdth per channel
- Higher-order modulation
- Multiple radios for transmitting multiple streams in parallel

# Measuring and Optimizing  
Popularity of WiFi created its biggest challenges
- Inter-cell interference
- Intra-cell interference

TODO fill out


Performance of WiFi
- No bandwidth/laatency guarantees
- Variable bandwidth based on envronment’s STN
- Transmit power limited to 200 mW and is likely less
- Limited amount of spectrum 
- Accesspoints overlap in chanel assingmnet by design
- Access points and peers compete for access to same radio channel



## Packet Loss in WiFi network
Probabilistic scheduling → high number of collisions between wireless peers
- Doesn’t necessarily = higher TCP packet loss- WFi protocols (physical + data layer implementations) have retransmission + error correction mechanisms


# Optimizing for WiFi networks

Hard to beat the simple convenience of wifi- let’s learn how we can leverage it in our WiFi networks

## Leverage Unmetered Bandwidth
In practice (usually)
- Wifi Network = extension to wired LAN
- Wired LAN connected via DSL, cable, or fiber

Most WiFi clients are sitll likely lmited by the WAN bandwidth, not WF itself

Typical wifi deployment backed by basically unmetered WAN connection

What does this mean? Large downloads/streaming/ updates are best done over WiFi!

#### Adapt to Variables Bandwidth
Remember- WiFi has no bandwidth or latency guarantees- alocation may change dramatically. Don’t forecast to far into the future

#### Adapt to Variable latency
Similarly- no guarantee on latency of first wireless hop

If applicatoin is latency sensitve, be very careful about adapting behavior when running overWifI network- may be goo reason to consider WebRTC,which offers the option of an unreliable UDP transport.