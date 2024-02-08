---
title: Optimizing Application Delivery
---

High performance browser networking is the sum total of each of the parts of our networking technogies

- Speed of light and distance dictate propagation latency 
- Medium (wired vs wireless) determines processing, transmission, queuing, and other delays incurred by each data packet

While wec an’t make bits travel faster, we can optimize transport and application layer optimizations to eliminate unenecseary rountrips, requestsand minimie distance traveled by each packet

API Layer
- Optimize resource downloads, beacons, etc.

Physical layer
- Ensure each server is using latest TCP + TLS best practices

Application layer
- Work around limitations of HTTP 1.x
- Learn to leverage performance nehamncements in HTTP 2.0
- Evergreen performance best practices

# Evergreen Performance best practices

Always try to 
1. Eliminate / reduce unnecesary latency 
2. Minimize amount of transfered bytes

Foundation for dozens of other familiar performance rules
- Reduce DNS Lookups
- Reuse TCP Connections
- Minimize number of HTTP redirects (ideally 0)
- Use a CDN
- Eliminate unnecessary resources

HTTP also has some mechanisms you can use (which are the following)
## Cache Resources on the Client

This lets you eliminate requests

For HTTP resources- set appropriate headers
- `Cache-Control` specificy cache lifetime of resource
- `Last-Modified` as validation mechanisms

When possible, specify an explicit cahce lifetime for each resource + validation mechanism


## Compress transferred data

Always apply best compression method for each asset you use
- Text-based can be reduced by 60-80% typically

Images, however-
- >1/2 of bytes avg.
- Can eliminate unnecessary metadata
- Resize on server
- Optimal image formats (e.g. webp)
- Use lossy compression

## Eliminate Unecessary Request Bytes
HTTP = stateless, server doesn’t need to retain information about client

Many applications require state (session management, personalization, analytics, etc.)

HTTP lets you associate/update cookie metadata for its origin (~4kb limit)

Monitor cookie size judiciously! ANd leverage.a shared sesion cache on teh server to look up other metadta

## Parallelize Request. Response Processing

To get fastest respone times, all resource requests should be dispatched as soon as possible.

Best performance best practices
- Connection keepalive + HTTP 1.0 → 1.1
- Multiple HTTP 1.1 when necesary for parllel downloads
- HTTP 1.1 pipelining
- Investigate HTTP 2.0
- Make sure server can process requests in parallel

# Optimizing for HTTP 1.x
TODO



