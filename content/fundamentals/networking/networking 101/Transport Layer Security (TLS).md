---
title: Transport Layer Security (TLS)
---
What is SSL?
- Protocol by netscape to enable ecomerce transaction
- Implemented on top of TCP, while alloweing protocols above it to operate unchanged
- End result of correct use: 3rd party observer can only infer 
	- Connection end-points 
	- Type of encryption
	- Frequency
	- Approximate amount of data sent
	- No modifying / reading actual data

What is TLS? Renaming of SSL (basically, SSL 3.0)

# Encryption, Authentication, and Integrity

TLS goal: provide 3 services to apps running above it:
- Encryption: obfuscate what’s sent
- Authenticatoin: verify validity of provided id material
- Data Integrity: detect message tampering

Requirements to establish secure data channel? (**encryption**)
- Agreement from connection peers on 
	- Ciphersuites
	- Keys to encrypt the data
- This is established by the TLS handshake

TLS Handshake: allows peers to negotiate shared secret key over unencrypted channel
- Also lets clients verify server identity (**authentication**)

Also, have MAC (message authentication code)
- Sign each message with it- 1-way cryptographic hash function
- Use to ensure **data integrity**

# TLS Handshake
In order for client / server to excahnge data oer TLS, must negotiate the encrypted tunnel
- TLS protocol version
- Ciphersuite
- Verify certifacites

The process (in order)
- Runs over TCP- so need to do the TCP 3 way handshake
- Client 
	- Sends: specifications in plain text (ie TLS protocol version, supported cipher-suites, other TLS options)
- Server
	- Decides: TLS protocol, ciphersuite- attaches certificate
	- Sends: response
- Client (if happy with certificate)
	- Generates: new symmetric key, encrypts with server’s public key- 
	- Sends: message to server, saying to switch to encrypted communication going forward
- Server
	- Decrypts: symmetric key sent by client, checks message integrity by verifying MAC- 
	- Sends: encrypted “Finished” message
- Client
	- Decrypts: message with symmetric key, verifies MAC
	- If all is well, tunnel is now established

Takeaways
- Complicated
- Requires TCP handshake + 2 extra round trips 

Most of this process is, luckily, handled by the server and browser. We just neeed to provide / configure certificates

## ALPN (Application Layer Protocol Negotiation)

How do you use a custom application protocols for network peer communication?

Slow/impractical solution: assign a well-known port to predetermined protocol and configure clients/servers as such.

Instead, we reuse port 443 (which is reserved for secure HTTPS sessions running over TLS)
- TLS gives reliability- end-to-end encrypted tunnel
- But what about how we negotiate the protocol? 
	- Could use HTTP upgrade mechanism
	- Better to use ALPN- avoid extra roundtrip

ALPN: TLS extension which introduces support for application protocol negotiation into the TLS handshake. How?
- Client adds `ProtocolNameList` field (list of support app protocols) into `ClientHello` message
- Server inspects the `ProtocolNameList` field and returns `ProtocolName` field indicated selected protocol as part of the `ServerHello` message

## Server Name Indication (SNI)

