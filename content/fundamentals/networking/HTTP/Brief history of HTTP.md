---
title: Brief history of HTTP Blocks of TCP
---
HTTP: one of most widely addopted application protocols on the internet
- Common language bewteen clients and servers


Let’s take a historical tour of the evoution of the HTTP protocol

# HTTP 0.9: one-line protocol

Original proposal: simplicity

Simple prototype with subset of proposed functionality

 General architecture
 - Client request: single ASCII character string
 - Client request: terminated by carriage return (CLRF)
 - Server response: ASCII character stream
 - Server response: HTML
 - Connection terminated afte document transfer is complete

But really- it just enables extremely smple Telnet ffriendly protocol

- Request- single line (e.g. GET and path)
- Response- single hypertext document


Recap of the features
- Client server, request-response protocol
- ASCII protocol, running over TCP/IP link
- Design to transfer HTML
- Connected between sever / client closed after every request

# HTTP 1.0 Grapid Growth and Informational RFC
1991-1995: rapid co-evolution of HTML / web browsers. + public internet infrastructure

Huge list of desired capabilties that exposed limitations of HTTP 0.9 
- Don’t want to serve just hypertext documents

So, May 1995 publish information RFC 1945- “common usage” of HTTP 1.0 implementations found in the wild

Key protocol changes
- Request can contain multiple newline separated header fields
- Response object prefixed with response status line
- Response object has own set of newline separated header fields
- Response object not limited to hypertext
	- Any content type- now is more like hypermedia transport
- Connection between server + client closed after every request


# HTTP 1.1 Internet Standard

Work on turning HTTP into official IETF internet standard
- First HTTP 1.1 standard in RFC 20168- January 1997

Resolved protocol ambiguites and introduced peformance optimizations
- Keepalive connections
- Chunks Encoding transfers
- Byte-range requests
- Transfer encodings
- Request Pipeling

Most obvious difference: connection keepalive allows reuse of existing TCP connection for multiple requests to the same host (ie image and hypertext)

Lots of other stuff going on


# HTTP 2.0 Improving Transport Performance
HTTP has taken over the world- all kinds of devices speak HTTP to power our lives

Has begun to show signs of stress, and sonew initative announced for HTTP2.0 in 2012.

Focus of HTTP 2.0
- Improving transport perforamnce
	- Lower latency
	- higher throughput
- None of the high-level protocol semantics affected (ie. http heaers, values, use case)

Whole separate sectin of HTTP 2.0

