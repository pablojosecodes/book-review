---
title: Primer on Web Performance
---

So far we’ve looks at individual networkign components in close detail,s but let’s look at end-to-end picture of web performance optimization
- Importance of latency and bandwidth
- TCP constaints imposed on HTTP
- Features/shortcomings of HTTP itsef
- Web app trends + performance requiements
- Browser constaints/optimizations

# Hypertext, web pages, and web apps

We’ve had 3 classes of experience
- Hypertet documents
	- Plain text with hyperlinks
- Web page
	- HTML to extend definittion of hyptertext for hypermedai
- Web apps
	- Javascript + DHTML + AJAX


back in HTTP 0.9 days, optimization was just trying to make single HTTP request over short-lived TCP connection as efficient as possible


Web pages: document → document + dependent resources
- Hence, HTTP headers in HTTP 1.0
- Also, primitives for performance like
	- Caching
	- Keepalive
	- Etc.
- Metric: document load → page load time

Web app: simple web page → compelx dependency graphs
- Now, metrics not just page load time → application specific 
	- Milestones in loading progress of application
	- TTI
	- Interactions of user
	- Engagement / conversion



# Anatomy of modern web apps

As of 2023 (average web app)
- HTML
	- Requests: 10
	- KB: 52
- Images
	- Requests: 55
	- KB: 812
- Javascript
	- Requests: 15
	- KB: 216
- CSS
	- Requests: 5
	- KB: 36
- Other
	- Requests: 5
	- KB: 195

## Human perception

These are all relative terms- we need user-centric perceptual processing time constants
- Instance: 0-100ms
- Small delay: 100-300ms
- Machine is working: 300-1000ms
- Menta context switch: 1s+
- Abandon: 10s+

## Resource waterfall

This is the single most insightful network diagnostic tool

Note
- While conrtent of document is being fetched, new HTTP requests dispatched
- HTML parsing is incremental- dispatch necessary requests in parallel (structure of markup matters)
- Rendering starts before all resources are loaded

Know what you’re optimizing for
- First render time?
- Document complete?
- Finished fetching last resource?

Can check out the connection view, which shows life of each TCP connection

Note that the botteneck for performance ofr most webapps isn’t bandwidth, but network roundtrip latency


# Performance Pillars: Computing, Rendering, Networking

3 primary tasks in executing web page
- Fetching rendering
- Page layout / rendering
- JS Execution


Why can’t we just wait for internet to get faster?

## More bandwidth doesn’t matter (much)

Access to higher bandwidth is alwas good, but roundtrip latency is limiting factor

Think of it as folllows
- Streaming HD video is bandwidth limited
- Loading / rendering homepage is latency limited


## Latency as a performance bottleneck

Latency generally matters much more as a bottleneck

Why? The performance characteristics of underlying protocols–
- TCP handshakes
- Flow / congestion control
- HOL blocking due to packe tloss

HTTP: small bursty data transfers

TCP: optimized for long-leved connection + bulk data

# Synthetic + Real-User Perforamnce Measurement
Remember: no single metric is perfect

And how to reliably gather perforamcne data

**Synthetic Testing**: controlled measurment environment
- Looks like one of these
	- Local build process runnig perforamcne suite
	- Load testing staging infra
	- Geo-distributed monitoring servers whih periodically perform set of actions
- Not sufficient for identifying all performance bottlenecks
- Not diverse enough
- Contributing factors are- we can’t simulate
	- User navigation patterns
	- User’ cache
	- Intermediate proxies/caces
	- Diverse user hardware
	- Diverse user browsers
	- Connectivity conditions

We must also measure real time user performnance, but how?


W3C introduced **Navigation Timing API**, widely supported
- Lots of data like DNS / TCP connect times

W3C also standardized **User Timing** API and **Resource Timing**
- User Timing: simple API for marking application specific performance metrics with high-resolution timers
- Resource Timing: Performance data for each resource

# Browsers Optimization

Browsers do a whole of of optimizing- of 2 types
- Document aware optimiation: Networking stack integrates with document/css/js parsing pipelines to dispatch critical network assets ars early as possible
- Speculative optimization: use knowledge of user’s navigation patterns

4 techniques employed by bowsers
- Resource pre-fetching and prioritization
- DNS pre-resolve
- TCP pre-connect
- Page pre-rendering

How to assist browser?
- Make critical resources discoverable as early as possible
- Deliver CSS as early as possible (unblocks renering/js execution)
- Defer noncritical JS so as to not block DOM and CSSOM construction
- Periodically flush document since HTML is parse incrementaly

Can also embed hints into the documentation