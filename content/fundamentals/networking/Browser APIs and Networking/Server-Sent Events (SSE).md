---
title: Server-Sent Events (SSE)
---
Purpose: enable efficient server-to-client streaming of text-based event data
- Examples: real-time notifications / updates generated on the server
- Features
	- Low latency delibery via single long-livd ocnnection
	-  Efficient browser message parsing with no unbounded buffers
	- Automatic tracking of last seen message and auto reconnect
	- Client message notifications as DOM event
- efficient cros browser implementation of XHR streaming, but browser handles conection management + message parsing

Basically- makes working with real time data simple and efficient

# EventSource API

EventSource interface
- Simple browser API (abstracts away the low level stuff!)

Example code
```typescript
var source = new EventSource("/path/to/stream-url");
source.onopen = function () { ... };
source.onerror = function () { ... };
source.addEventListener("foo", function (event) {
	processFoo(event.data);
});
							  
source.onmessage = function (event) {
	log_message(event.id, event.data);
	if (event.id == "CLOSE") {
		source.close();
	}
}
```
1. Open new SSE connection to stream endpoint
2. Optional callback, invoked when connection is established 
3. Optional callback, invoked if the connection fails
4. Subscribe to event of type “foo”; invoke custom logic Subscribe to all events without an explicit type
5. Close SSE connection if server sends a “CLOSE” message ID

That’s it for the client API! Handles a lot for us. App just focuses on:
- Open new connection
- Process event notifications
- Terminate stream when finished

But- how does the browser know the [] of each message?
- ID
- Type
- Boundary

Event stream protocol!

# Event Stream Protocol

SSE stream is delivered as a streaming HTTP response
- Client: initiates regular HTTP request
- Server: custom “text/event-stream” content-tpe and then streams UTF-8 encoded event stream

# SSE Use Cases and Performance

Recap: SSE is a high-performance transport for server-to-client streaming of text-based real- time data.
- minimal message overhead
- messages can be pushed the moment they’re available on the server

2 key limitations
- Server-to-client only- can’t address the requeset streaming use case
- Specifically designed to tansfer UTf-8 data- not efficient for binary streaming
	- Can solve this at the application layer
