---
title: Primer on Browser Networking
---
Modern browser is designed for efficiency and security

Networking stack is complex with its own optimization criteria, APIs, and services

Browser worries about individual TCP / UDP sockets for us, as well as right connection limits, formatting requests, etc.

But out of site != out of mind. Understanding optimization can help with performance

# Connection Management and Optimization

Web apps in browser don’t manage lifecycle of individual network sockets
- Separation betwee request managment and socke tmangement

Socket 
- In pools 
- Grouped by origin
	- Origin- triple (protoco, domain name, port number)
- Each Pool
	- Own conection limits / security constraints


Automatic socket pooling
- Automates TCP rconnection reuse
- Enables
	- Serving queued requests in priority order
	- Reusing of sockets to minimize latency / improve throughput
	- Opening scokets anticipatorily
	- Optimize idle socket closing
	- Optimizing bandwidth

# Network Security + Sandboxing

Deferring individual socke tmanagmetn also alows sandboxing / consistent security / prolicy constraints (no malicious application code!)


Some restrictions in place
- Connection limits: 
	- Browser manages all open socket pools + enforces connection limits
- Request formatting + response processing
	- Browser formats all requests + decodes automatically
- TLS negotiation
	- Performs handshake = verification checks
- Enforces same-origin policy
- etc.

# Resource + client state caching

Remember- fastest request is one not made!

- Browser automatically checks resource cache before sending out request
- Also, browser does auth / session / cookie management

# Application APIs and protocols
Now we’re finally at the application APIs and protocols

When we do anything from initiate HTTP request, a websocket sesion, or WebRTC, we are interacting with some or all of the underlying services we’ve talked about at the lower layesrs.

No best protocol / API- will need to use a mix of transports based on variety of requiremetsnts

Some high level features of XHR, SSE, and WebSocket
TODO