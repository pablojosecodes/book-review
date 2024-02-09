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



