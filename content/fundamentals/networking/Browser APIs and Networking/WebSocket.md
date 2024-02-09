---
title: WebSocket
---
Functional summary
- Bidirectiona message oriented streaming (client - server)
- Text + binary data
- Closest thing to a raw network socket in the browser

Additioanl services
- Connection negotiation
- Interoperatibility with existing HTTP infra
- Message-oriented communication and efficient message framing
- Subprotcool negotiation and extensivility

# The WebSocket API
Small + simple!


Example connection initiation
- Mostly just need URL of websocket resources and a few application callabacks

```typescript
var ws = new WebSocket('wss://example.com/socket');
    ws.onerror = function (error) { ... }
    ws.onclose = function () { ... }
    ws.onopen = function () {
      ws.send("Connection established. Hello server!");
}
    ws.onmessage = function(msg) {
      if(msg.data instanceof Blob) {
        processBlob(msg.data);
      } else {
        processText(msg.data);
      }
}
```
The general order
1. New Websocket connection
2. Optional callback (when connection error)
3. Optional callback (when connection end)
3. Optional callback (when connection established)
4. Client initiated message to the server
5. Callback funtion for each new message
6. Invoke binary or text processing logic for received message

Pretty much self-explanatory


## WS and WSS URL Schemes

Looking at API, uses 
- `ws` for plain text
- `wss` when need encrypted channel

Why not use `http`?
- WebSocket is primarily for brwoser - server communication cchannels, but can be used outside browser and negotiated via non-HTTP exchange

## Receiving Text and Binary data

WebSocket doesn’t need to worry about buffering parsing, reconstructing received data.
- `onmessage` callback is only called when the entire message is availale on the client

When browser receives new message, automatically converts it to a _  then passes directly to the application
- DOMString object for text-based data
- Blob object for binary data
- Can also convert binary data to ArrayBuffer instead

>Note: Blob is a file-ike object of immutable raw data. ArrayBuffer is likely better if you want to modify and whatnot.

## Sending Text and Binary Data

Once you establish websocket connection client can send / receive UTF-8 a d binary emessages at will

Example code
```typescript
var ws = new WebSocket('wss://example.com/socket');
ws.onopen = function () {
	socket.send("Hello server!");
	socket.send(JSON.stringify({'msg': 'payload'}));
	var buffer = new ArrayBuffer(128);
	socket.send(buffer);
	var intview = new Uint32Array(buffer);
	socket.send(intview);
	var blob = new Blob([buffer]);
	socket.send(blob);
}
```
1. Send UTF-8 encoded text message
2. Send UT-8 encoded JSOn paload
3. Send Arraybuffer conents as binary payload
4. Send the ArrayBufferView contents as binary payload
5. Send the Blob contents as binary payload

The websocket api accepts DOMstring object (encoded as UTF-9) or (for binary transfer) either Arraybuffer, ArrayBufferView, Blob 

`send()` is asynchornous
- Data is queued vy clienta nd function returns immediately- so data is not immediatley sent!!
- But, you can monitor how much data has been queued by the browser
	- Subscribe to update, check amount of buffered data on client, send update if buffer is empty
```typescript
ws.onopen = function () {
	  subscribeToApplicationUpdates(function(evt) {
		if (ws.bufferedAmount == 0)
		  ws.send(evt.data);
	}); 
};
```

Why bother checking if previous messages have ben drained from the lcient’s buffer?
- Help avoid head of line blocking


## Subprotocol Negotiation

WebSocket makes no assumption about format of each message
- Just a single bit tracks (text or binary?)

And no additional metadata like HTTP headers
- What to do if you want additional metadata? You’ll need to agree to implement a subprotocol

Luckily, WebSocket offers a subprotocol negotiation API 
- Clients can advertise which protocols it supports to the server as part of its initial connection handshake.

How?
```typescript
 var ws = new WebSocket('wss://example.com/socket',
                           ['appProtocol', 'appProtocol-v2']);

```
Then to check which ones the server chose
```typescript
ws.onopen = function () {
  if (ws.protocol == 'appProtocol-v2') {
  ...
```
# WebSocket Protocol

2 high level components
1. Opening HTTP handshake
	1. Negotiates paramters of ocnnection
2. Binary message framing mechanism
	1. Low overhead message based delivery of text / binary data

## Binary Framing Layer

TODO


# WebSocket Use Cases and Performance

Again, the WebSocke tAPI provides a simple interface for bidirectional message-oriented stremaing of text/binary data between client and server.

Super simple set up:
1. WebSocket URL
2. A few JS callback functions
3. Up and running

And WebSocket Protocol
- Cusotm application protocols enabled!

Still important to understand the implementation details to optimize our applications!

## Request and Response Streaming

