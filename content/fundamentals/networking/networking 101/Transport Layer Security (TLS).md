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
- Client sends specifications in plain text (ie TLS protc)