---
title: WebRTC
---

WebRTC
- Collection fo standards, protocls, and JS APIs
- Enables P-to-P audio/video/daa shareing between browsers
- Leverage via Javascript API
- Transports data over UDP


3 primary APIs
- `MediaStream`- acquisition of audio/video streams
- `RTCPeerConnection`- commnicating audio/video data
- `RTCDataChannel`- communicating arbitrary app data

# Standards and Development
Real-time communication is ambitious!

WebRTC architecture has over a dozen standards
- WEBRTC: responsible for browser APIs
- RTCWEB- responsible for other things (protocols, data formats, security, etc.)

Can integrate with existing communication systems
- OIP, SIP clients, een PSTN


# Audio and Video Engines
Hard requirements
- Browser: access system hardware to capture raw audio and process it to accuont for network challenges
	- All done directlyin the browser
- Client: decode streams in real time, adapt to network conditions

## Acquiring Audio / Video with `getUserMedia`
Remember that the `MediaStream` object is the primary interface for functionality of capturing media.

`MediaStream` object: represents a real time media stream
- 1+ tracks (`MediaStreamTrack`)
- Tracks within are synchronizedd
- Input: microhone, webcam, local/remote file
- Output sent to: local video/audio element, JS for processing, remote peer

Example usage
```typescript
 <video autoplay></video>
    <script>
      var constraints = {
        audio: true,
        video: {
          mandatory: {
            width: { min: 320 },
            height: { min: 180 }
          },
          optional: [
            { width: { max: 1280 }},
            { frameRate: 30 },
            { facingMode: "user" }
] }
}
      navigator.getUserMedia(constraints, gotStream, logError);
      function gotStream(stream) {
        var video = document.querySelector('video');
        video.src = window.URL.createObjectURL(stream);
}
      function logError(error) { ... }
    </script>
```
Here we see
- HTML video output
- Request
	- Mandatory audio
	- Mandatory video
- List of mandatory constraints + array of optional constraints
- Request audio/video streams
- Callback function- process Media Stream

Once stream is acquired- can feed to other browser APIs
- Web Audio API- process in browser
- Canvas API- post process individual video frames
- CSS3 + WebGL APIs- apply 2d/3d effects

# Real-time Network Transports

Timeliness matters a lot for real-time communication- we should be able to handle intermittent packet loss
- This is why UDP is preferred transport for deivering real-time data

> Remember: [[Building Blocks of UDP|UDP’s]] non-services

But we also need protocols and services above UDP
- ICE
- SDP
- DTLS
- SCTP
- SRTP

## Briefly on `RTCPeerConnection`

What it does: manages peer-to-peer connection
- Manage ICE workflow for NAT traversal
- Send automatic STUN keepalives between peers
- Keep track of local + remote streams
- Trigger automatic stream renegotiation when required
- Provide APIs for generating connection offer, accepting answer, query connection for current statet, etc.


But to understand this, we need to understand singaing/negotiation, offer-answer workflow, and ICE traversal- let’s talk about that now

# Establishing a peer-to-peer conenection
Sadly, opening peer-to-peer connection requires a lot more than ust opening an XHR, EventSource, or WebSocket session
- These rely on well-defined HTTP handshake and assume desintatin serve is reachable by client
- In contrast: two WebRTC peers within own distrinct private nwtorks (likely) and behind layers of NAT

Initiating sessions
- Gather IP/post candidates, traverse NATs, then run connectivity checks

But! When we open HTTP connection to server, you’re assuming that hte serve is listening for handshaek- but this isn’t necessairly true. This means we need to
1. Notify other peer of intent o opern peer-to-peer connection
2. Identify potential routing paths
3. Exchange info about paramters of media/data streams

### Signaling and Session Negotiation



First step- is peer reachable and willing to establish connection?

But, how do you notify peer if its not listening for incoming packets?

Well, we need at least a shared signaling channel

WebRTC lets application deicde the choice of signaling transport and protocol
- Allows for interoperability with other signaling protocols that aready power comunication infrastructure, or alternatively implement custom protocol for signaling service

> Example of custom signaling: Skype. users ned to connect to their signaling servers (with proprietary protocol) to initiate peer-to-peer connection

### Session Description Protocol

Assuming application implements shared signaling channel, can perform first stpes for initating a WebRTC connection

```typescript
var signalingChannel = new SignalingChannel();
    var pc = new RTCPeerConnection({});
    navigator.getUserMedia({ "audio": true }, gotStream, logError);
    function gotStream(stream) {
		pc.addstream(stream);
		pc.createOffer(function(offer) {
		pc.setLocalDescription(offer);
		signalingChannel.send(offer.sdp);
      });
}
```

How it works
1. Initialize the shared signaling channel
2. Initialize the RTCPeerConnection object
3. Request audio stream from the browser
4. Register local audio stream with RTCPeerConnection object Create SDP (offer) description of the peer connection
5. Apply generated SDP as local description of peer connection Send generated SDP offer to remote peer via signaling channel

WebRTC uses SDP (Session descriptoin Protocol) to esribe parametesr of the peer-to-peer connection
- Describes session profile (list of properties of the connection)
- Simple and text-based 
- WebRTC apps don’t need to deal with it directly- just need to cal a few functions on the object
- In the above, the `createOffer()` generates the SDP description of hte intended sesion

The SDP itself looks like this
```
...
m=audio 1 RTP/SAVPF 111 ...
a=extmap:1 urn:ietf:params:rtp-hdrext:ssrc-audio-level
a=candidate:1862263974 1 udp 2113937151 192.168.1.73 60834 typ host ...
a=mid:audio
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10
...
```
1. Secure audio profile with feedback
2. Candidate IP, port, and protocol for the media stream
3. Opus codec and basic config

Once the offer is generated, can be sent to remote peer via the signaling channel (encoding up to the application)- followiwng a symmetric workflow
1. Initiator 
	1. Registersone or more streams with `RTCPeerConnection`
	2. Creates offer
	3. Sets it as “local description” of the session
	4. Sends generated session offer to other peer
2. Once offer is recieved, receiever 
	1. Sets the initiatior’s’ description as the “remote description” of the session
	2. Registers his own streams with his own `RTCPeerConnection`
	3. Generates “answer” SDP description  and sets as the local description o the session
	4. Sends generated session answer back to initiator
3. Once answer received by initiator, sets receiver’s answer as the “remote description” of original session


Now: SDP session descriptions have been exchanged via the signaling channel and have negotiated the ettings of streams, just need to deal with connectivity checks and NAT traversal

### Interactive Connectivity Establishment (ICE)

Kind of tautologically: peers need to be able to route packets to each other (for peer-to-peer connection)

But how? Lots of firewalls and NAT devices between most peers

Trivial Case: both peers on same internal network, no firewalls, no NATs
- Peer can query OS for IP address
- Appen IP and port tuples to generated SDP strings
- Forward to other peers
- Once SDP exchange compllete, can initiate direct P-to-P connection

But- what about distinct private networks?
- Issue: We need a public routing path between the peers
- Solution: WebRTC framework manges most of this on our behalf- each `RTCPeerConnection` has an “ICE agent”:
	- Gathers local IP, port tuples (candidates)
	- Performs connectivity checks b/w peers
	- Sends connection keepalives
- Once session is set, local ICE agent automatically beigins discovering posible candidates for the local peer
	- Queries OS for local IP addresses
	- If configured- queries external STUN server to retrieve public IP and port tupel of per
	- if configured- appends TURN server as a last resort candidate
- When new candidate is discovered, agnet registers with the `RTCPeerConnection` object and notifies applicatio nvia callback function


Extending our earlier example to work with ICE
```typescript
var ice = {"iceServers": [
  {"url": "stun:stun.l.google.com:19302"},
  {"url": "turn:user@turnserver.com", "credential": "pass"}
]};
var signalingChannel = new SignalingChannel();
var pc = new RTCPeerConnection(ice);
navigator.getUserMedia({ "audio": true }, gotStream, logError);
function gotStream(stream) {
	pc.addstream(stream);
	pc.createOffer(function(offer) {
		pc.setLocalDescription(offer);
	}); 
}

pc.onicecandidate = function(evt) {
	if (evt.target.iceGatheringState == "complete") {
		local.createOffer(function(offer) {
			console.log("Offer with ICE candidates: " + offer.sdp);
			signalingChannel.send(offer.sdp);
		}); 
	}
} 
...
// Offer with ICE candidates:
// a=candidate:1862263974 1 udp 2113937151 192.168.1.73 60834 typ host ...
// a=candidate:2565840242 1 udp 1845501695 50.76.44.100 60834 typ srflx ...
```

You can see that ICE agent handles most hte complexity on our behalf

## Incremental Provisioning (Trickle ICE)
ICE gathering process takes a while

TODO



## Delivering Media and Application Data

Halfways through the WebRTC protocol stack when weve establishe da peer-to-peer connection. Now we have
- Raw UDP connecctions open to each other

We’re missing encryption of communication and efficient data fow
- DTLS: negotiate the secret keys for encrypting media data
- SRTP: transport audio and video streams
- SCTP: transport application data


## Secure Communication with DLTS

WebRTC requires all transferred dat be encrypted whie in transit

Why not use TLS?
- TLS can’t be used over UDP- TLS requires on the features of TCP

Instead, use DTLS, which is realy just TLS with minimal modifications for compatibility with datagram transport offered by UDP

DTLS addresses the following issues with TLS
- Requires reliable, in order, fragmentation friendly deiliver of handshake records to negotiate the tunnel
- TLS integrity checks may fail if records are
	- Fragmented across multiple packets
	- Processed out of order

How does DTLS do this?
- Implements a mini-TCP for the handshake sequence
- Adds 2 rules
	- DTLS rcords must fit into single network packet
	- Block ciphers (not stream ciphers) must be used for encrypting record data

## Delivering Media with SRTP and SRTCP

WebRT- media acquisitino and delier y asfully managed service
Applciation does NOT have any control over video’s optimization

How does WebRTC optiize and adapt quaality of each media stream? Reuses existing trasnport protocls used by VoIP phones, communication gateways,etc.

**Secure Real-time Transport Protocol** (SRTP): Secure profile of the standardized format for delivery of real-time data, such as audio and video over IP networks.
- Defines standard packet format for delibering auio and video over IP networks
- No mechanism / guarantees on timeliness / reliability, etc
- Each SRTP packet
	- Has an auto-incrementing sequence number
	- Carries a timestamp
	- Carries an SSRC identifier
	- May contain optional metadata
	- Carries encrypted media payload and an auth tag
- Track snumber of sent / lost bytes / packets and periodically adjusts sending rate / parameters

**Secure Real-time Control Transport Protocol** (SRTCP): Secure profile of the control protocol for delivery of sender and receiver statistics and control information for an SRTP flow.


## Delivering Applicatoin data with SCTP

WebRTC also allows P-t-P trasnfers of arbitrary application data via DataChannel API


Why not use SRTP protocol?
- Not suitable for application data, we need to define the SCTP protocol

First, what are the `WebRTC` requirements for the `RTCDataChannel` interface and transport protocol
- Support multiplexing of multile channels
	- In-order or out of order dlelivery
	- Reliable / unreliable delivery
	- Priority level for each channel
- Message-oriented API
	- May be fragmented / reassembled
- Flow / congestion control
- Confidentialtity + integrity of transferred data

SCTP provides the best features of TCP and UDP  (lots of stuff within this protocol)
- Some terms:
	- Association: basically, a connection
	- Steram: unidrectional channel (sequential delivery unless confidgured for unordered)
	- Message: application data submitted to the protocol
	- Chunk: smalles unit of communication within SCTP packet
- Single SCTP association: may carry mutliple independnt sterams 
	- Each of ach which communicates by transferring application messages
		- Messages may be slit into chunks
- SCTP Packet
	- Common header
	- One or more control data chunks (lots of stuff within this chunk)
- How does it negotiate starting parametrs ofr the association?
	- Requires handshake sequence
- In sum- provides simlar services as TCP but since its tunneled ove UDp and implemented by WebRTC client, offers much more powerful API


# DataChannel

What does DataChannel do?
- Bidirectional exhcange of arbitrary applciation data wbtween peers
- Intuition: WebSocket but peer to peer and customizable delibery properrties of the underlying transport

Sample code
```typescript
function handleChannel(chan) {
	chan.onerror = function(error) { ... }
	chan.onclose = function() { ... }
	chan.onopen = function(evt) {
		chan.send("DataChannel connection established. Hello peer!")
	}
	chan.onmessage = function(msg) {
		if(msg.data instanceof Blob) {
			processBlob(msg.data);
		} else {
			processText(msg.data);
		}
	}
}
									
var signalingChannel = new SignalingChannel();
var pc = new RTCPeerConnection(iceConfig);
var dc = pc.createDataChannel("namedChannel", {reliable: false});
...
handleChannel(dc);
pc.onDataChannel = handleChannel;
```

1. Register WebSocket like callbacks on DataChannel object
2. Initialize new DataChannel with best-effort delivery semantics
3. Regular RTCPeerConnection offer/answer code
4. Register callbacks on locally initialized DataChannel
5. Register callbacks on DataChannel initiated by remote peer

DataChannel intentionally mirrors WebSocket
TODO